"""
agent.py — NeuroClaw_Bot: Local AI Coding Agent (Dual-Model Edition)

Runs TWO local models simultaneously:
  🧠 Phi-3-mini-4k-instruct  → Planning, reasoning, explaining
  ⚙️  Qwen2.5-Coder-3B        → Code writing, editing, debugging

First time setup:
    python setup_models.py          ← Downloads both models (~4.2 GB)

Then run inside any project folder:
    cd my-project
    python /path/to/local-claw/agent.py

Telegram remote control (optional):
    python agent.py --telegram
"""
import os
import sys
import argparse
from pathlib import Path

# Bootstrap: add local-claw directory to Python path
AGENT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(AGENT_DIR))


# ── Argument Parser ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="NeuroClaw_Bot",
        description="Local AI Coding Agent with dual-model support (Phi-3 + Qwen-Coder).",
    )
    parser.add_argument("--telegram", action="store_true",
                        help="Start in Telegram remote control mode.")
    parser.add_argument("--coder-model", type=str, default=None,
                        help="Override path to the coder model (.gguf).")
    parser.add_argument("--planner-model", type=str, default=None,
                        help="Override path to the planner model (.gguf).")
    parser.add_argument("--single-model", type=str, default=None,
                        help="Use a single .gguf model for both roles.")
    parser.add_argument("--gpu-layers", type=int, default=-1,
                        help="GPU layers to offload (-1 = auto, 0 = CPU only).")
    parser.add_argument("--ctx", type=int, default=4096,
                        help="Context window size in tokens.")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force rebuild of the project vector index.")
    parser.add_argument("--two-phase", action="store_true",
                        help="Use two-phase mode: Phi-3 plans, Qwen-Coder executes.")
    parser.add_argument("--advanced", action="store_true",
                        help="Experimental: Use Advanced Multi-Agent Orchestrator with file-editing tools.")
    return parser.parse_args()


# ── Config Loader ──────────────────────────────────────────────────────────────
def load_config() -> dict:
    config_path = AGENT_DIR / "config.yaml"
    if config_path.exists():
        try:
            import yaml  # type: ignore
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
    return {}


# ── Model Loading ──────────────────────────────────────────────────────────────
def load_models(args: argparse.Namespace, config: dict):
    """
    Load Phi-3 (planner) and Qwen-Coder (coder) models.
    Returns (planner_loader, coder_loader, router)
    """
    from model.loader import ModelLoader
    from model.model_router import ModelRouter
    from core.system_optimizer import get_optimal_threads, get_optimal_gpu_layers, get_optimal_ctx

    base_gpu = args.gpu_layers if args.gpu_layers != -1 else config.get("n_gpu_layers", -1)
    if base_gpu == -1:
        gpu = get_optimal_gpu_layers()
    else:
        gpu = base_gpu

    optimal_threads = get_optimal_threads()
    optimal_ctx = get_optimal_ctx()

    # Strictly cap context to optimizer's limit 
    ctx = min(args.ctx or config.get("planner_ctx", 4096), optimal_ctx)

    models_dir = str(AGENT_DIR / "models")

    # ── Single model override ───────────────────────────────────────────────
    if args.single_model:
        print(f"[Setup] Single-model mode: {Path(args.single_model).name}")
        loader = ModelLoader(models_dir=models_dir, n_ctx=ctx, n_gpu_layers=gpu, n_threads=optimal_threads)
        loader.load(model_path=args.single_model)
        router = ModelRouter(planner_loader=loader, coder_loader=loader)
        return loader, loader, router

    # ── Resolve model paths from args or config ─────────────────────────────
    coder_path = (
        args.coder_model
        or _resolve_model_path(config.get("coder_model"), models_dir)
    )
    planner_path = (
        args.planner_model
        or _resolve_model_path(config.get("planner_model"), models_dir)
    )

    planner_loader = None
    coder_loader = None

    # ── Load Qwen2.5-Coder (coder) ──────────────────────────────────────────
    print("\n" + "─" * 50)
    print("⚙️  Found CODER model (Qwen2.5-Coder-3B). Real-time loading enabled.")
    coder_loader = ModelLoader(
        models_dir=models_dir,
        n_ctx=config.get("coder_ctx", ctx),
        n_gpu_layers=gpu,
        n_threads=optimal_threads,
    )
    coder_path = coder_path or _find_coder_model(models_dir)
    coder_loader._model_path = coder_path # Assign path but DO NOT .load() yet

    # ── Load Phi-3-mini (planner) ───────────────────────────────────────────
    print("─" * 50)
    print("🧠 Found PLANNER model (Phi-3-mini). Real-time loading enabled.")
    planner_loader = ModelLoader(
        models_dir=models_dir,
        n_ctx=config.get("planner_ctx", ctx),
        n_gpu_layers=gpu,
        n_threads=optimal_threads,
    )
    planner_path = planner_path or _find_planner_model(models_dir)
    planner_loader._model_path = planner_path # Assign path but DO NOT .load() yet

    # ── At least one model required ─────────────────────────────────────────
    if not coder_loader and not planner_loader:
        print("\n❌ No models found. Run first:\n  python setup_models.py\n")
        sys.exit(1)

    if not coder_loader:
        print("  ℹ  No coder model — using planner for all tasks.")
    if not planner_loader:
        print("  ℹ  No planner model — using coder for all tasks.")

    router = ModelRouter(planner_loader=planner_loader, coder_loader=coder_loader)

    print("\n✅ Model Router ready:")
    for role, name in router.available_models.items():
        emoji = "⚙️ " if role == "coder" else "🧠"
        print(f"   {emoji}  {role.upper()}: {name}")

    return planner_loader, coder_loader, router


def _resolve_model_path(path_str: str | None, models_dir: str) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return str(p)
    # Try relative to models_dir
    alt = Path(models_dir) / path_str
    if alt.exists():
        return str(alt)
    return None


def _find_coder_model(models_dir: str) -> str:
    """Find the Qwen coder model in models/."""
    for pat in ["qwen", "coder", "deepseek"]:
        for f in Path(models_dir).glob("*.gguf"):
            if pat in f.name.lower():
                return str(f)
    raise FileNotFoundError("No coder .gguf file found in models/")


def _find_planner_model(models_dir: str) -> str:
    """Find the Phi-3 or general model in models/."""
    for pat in ["phi", "mistral", "llama", "gemma"]:
        for f in Path(models_dir).glob("*.gguf"):
            if pat in f.name.lower():
                return str(f)
    # Fall back to any remaining gguf
    files = list(Path(models_dir).glob("*.gguf"))
    if files:
        return str(files[0])
    raise FileNotFoundError("No planner .gguf file found in models/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    config = load_config()
    project_root = os.getcwd()

    print(f"\n{'='*60}")
    print(f"  🦞 NeuroClaw_Bot — Dual-Model Local AI Coding Agent")
    print(f"  Project : {project_root}")
    print(f"{'='*60}")

    # 1. Load both models
    planner_loader, coder_loader, router = load_models(args, config)

    # 2. Build / Load RAG Index
    from rag.indexer import ProjectIndexer
    indexer = ProjectIndexer(project_root)
    if not args.rebuild_index:
        loaded = indexer.load_index()
    else:
        loaded = False
    if not loaded:
        print("\n[RAG] Building project index...")
        indexer.build_index()

    # 3. Initialize agent components
    from tools.registry import ToolRegistry
    from core.memory import ConversationMemory
    from core.agent_loop_dual import DualModelAgentLoop

    tools = ToolRegistry(project_root)
    memory = ConversationMemory(max_tokens=2000)
    agent = DualModelAgentLoop(
        router=router,
        tools=tools,
        memory=memory,
        max_iterations=config.get("max_iterations", 12),
        two_phase=args.two_phase,
    )

    # 4. Start interface
    if args.telegram:
        _run_telegram(agent, config)
    else:
        _run_cli(agent, indexer, router, args, project_root)


# ── CLI REPL ───────────────────────────────────────────────────────────────────
def _run_cli(agent, indexer, router, args, project_root):
    from model.model_router import ModelRole
    
    # Init Orchestrator if in advanced mode
    orchestrator = None
    if args.advanced:
        from core.orchestrator import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator(router, workspace_path=project_root)
        print("\n🔥 ADVANCED MODE ENABLED: Agent can now execute multi-step tool workflows. 🔥")

    print("\n" + "─" * 60)
    print("✅ Agent ready! Type a task or a special command.\n")
    print("  Special commands:")
    print("  • clear       — Clear conversation memory")
    print("  • reindex     — Rebuild the project vector index")
    print("  • models      — Show loaded models")
    print("  • use coder   — Force next task to use the coder model")
    print("  • use planner — Force next task to use the planner model")
    print("  • quit / exit — Exit")
    print("─" * 60 + "\n")

    force_role = None

    while True:
        try:
            task = input("You ▸ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not task:
            continue

        # ── Special commands ───────────────────────────────────────────────
        if task.lower() in ("quit", "exit", "q", "bye"):
            print("Goodbye! 👋")
            break
        if task.lower() in ("clear", "reset"):
            agent.memory.clear()
            print("[Memory cleared]")
            continue
        if task.lower() == "reindex":
            print("[Reindexing project...]")
            indexer.build_index(force_rebuild=True)
            continue
        if task.lower() == "models":
            print("[Loaded models]")
            for role, name in router.available_models.items():
                emoji = "⚙️ " if role == "coder" else "🧠"
                print(f"  {emoji} {role}: {name}")
            continue
        if task.lower() == "use coder":
            force_role = ModelRole.CODER
            print("[Next task will use the CODER model]")
            continue
        if task.lower() == "use planner":
            force_role = ModelRole.PLANNER
            print("[Next task will use the PLANNER model]")
            continue

        # ── Enrich with RAG context ────────────────────────────────────────
        relevant = indexer.retrieve(task, top_k=3)
        if relevant:
            context = indexer.format_context(relevant)
            enriched = f"{task}\n\n[Project context from RAG]\n{context}"
        else:
            enriched = task

        if args.advanced and orchestrator is not None:
             orchestrator.run_advanced_task(enriched)
             print("\n[Return to Main REPL]")
        else:
             result = agent.run(enriched, force_role=force_role)
             force_role = None  # reset after use
             model_tag = f"[{router.last_used_model}]"
             print(f"\n🤖 Agent {model_tag} ▸ {result}\n")


# ── Telegram Gateway ───────────────────────────────────────────────────────────
def _run_telegram(agent, config: dict):
    from gateway.telegram_gateway import TelegramGateway
    token = config.get("telegram_token") or os.environ.get("TELEGRAM_BOT_TOKEN")
    user_ids_raw = config.get("telegram_allowed_user_ids") or os.environ.get("TELEGRAM_USER_IDS", "")
    if not token:
        print("❌ No Telegram token! Set telegram_token in config.yaml")
        sys.exit(1)
    allowed_ids = [int(str(uid).strip()) for uid in (user_ids_raw if isinstance(user_ids_raw, list) else str(user_ids_raw).split(",")) if str(uid).strip().isdigit()]
    if not allowed_ids:
        print("❌ Set telegram_allowed_user_ids in config.yaml")
        sys.exit(1)
    TelegramGateway(token=token, agent=agent, allowed_user_ids=allowed_ids).run()


if __name__ == "__main__":
    main()
