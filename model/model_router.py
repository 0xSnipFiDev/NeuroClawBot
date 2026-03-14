"""
model/model_router.py — Dual-model routing for NeuroClaw_Bot.

Inspired by Claude Code's multi-model architecture where:
- A FAST/SMALL model handles planning and reasoning (Phi-3-mini)
- A CODER model handles code generation and file edits (Qwen2.5-Coder-3B)

The router automatically picks the right model based on the task keyword
analysis, just like Claude Code routes between Haiku and Sonnet.

Model Roles
-----------
PHI-3-MINI  : General reasoning, planning, explaining code, answering questions
QWEN-CODER  : Writing code, editing files, debugging, generating functions

Usage
-----
    router = ModelRouter(phi_loader, qwen_loader)
    result = router.generate(task="Fix the bug in main.py", prompt=full_prompt)
    print(f"Used model: {router.last_used_model}")
"""

from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.loader import ModelLoader


class ModelRole(str, Enum):
    CODER = "coder"     # Qwen2.5-Coder-3B — code writing / editing / debugging
    PLANNER = "planner" # Phi-3-mini — planning / reasoning / explaining


# ── Keyword sets for task routing ─────────────────────────────────────────────
# If the task contains any CODER keywords → use Qwen Coder
CODER_KEYWORDS = {
    "write", "create file", "edit", "modify", "refactor", "implement",
    "fix bug", "debug", "add function", "add method", "add class",
    "generate", "code", "function", "class", "def ", "import",
    "unittest", "pytest", "test", "fix error", "syntax error",
    "type hint", "docstring", "write_file", "the code", "program",
    "script", "algorithm", "loop", "api", "endpoint", "database",
    "sql", "html", "css", "javascript", "typescript",
}

# If the task contains any PLANNER keywords → use Phi-3
PLANNER_KEYWORDS = {
    "explain", "what is", "how does", "describe", "summarize",
    "analyze", "review", "understand", "list", "find", "search",
    "show", "read", "tell me", "what are", "plan", "suggest",
    "architecture", "structure", "overview", "why", "check",
    "compare", "difference", "help", "advice",
}


class ModelRouter:
    """
    Routes generation calls to either the Planner (Phi-3-mini) or
    Coder (Qwen2.5-Coder-3B) model based on task content analysis.

    Falls back gracefully if only one model is loaded.
    """

    def __init__(
        self,
        planner_loader: "ModelLoader | None" = None,
        coder_loader: "ModelLoader | None" = None,
    ):
        self.planner = planner_loader   # Phi-3-mini
        self.coder = coder_loader       # Qwen2.5-Coder-3B
        self.last_used_model: str = "none"
        self.last_used_role: ModelRole | None = None

        if not planner_loader and not coder_loader:
            raise ValueError("At least one model must be provided to ModelRouter.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def select_role(self, task: str) -> ModelRole:
        """
        Decide which model role is best for a given task string.

        Logic:
        1. Count CODER keyword hits vs PLANNER keyword hits
        2. If CODER wins or tie → use CODER (coding is primary function)
        3. If PLANNER wins clearly → use PLANNER
        """
        task_lower = task.lower()
        coder_score = sum(1 for kw in CODER_KEYWORDS if kw in task_lower)
        planner_score = sum(1 for kw in PLANNER_KEYWORDS if kw in task_lower)

        if coder_score >= planner_score:
            return ModelRole.CODER
        return ModelRole.PLANNER

    def generate(
        self,
        prompt: str,
        task: str = "",
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
        force_role: ModelRole | None = None,
    ) -> str:
        """
        Generate text, routing to the right model.

        Args:
            prompt: The full formatted prompt string.
            task: The original user task (used for routing decisions).
            force_role: Override automatic routing.

        Returns:
            Generated text string.
        """
        role = force_role or self.select_role(task)
        loader = self._pick_loader(role)

        self.last_used_model = loader.model_name
        self.last_used_role = role
        role_emoji = "⚙️ " if role == ModelRole.CODER else "🧠"
        print(f"[Router] {role_emoji} Using {role.value} model → {loader.model_name}")

        return loader.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

    def generate_with_both(
        self,
        planning_prompt: str,
        coding_prompt: str,
        max_tokens: int = 512,
    ) -> tuple[str, str]:
        """
        Two-phase generation:
        1. Phi-3-mini creates a plan
        2. Qwen-Coder executes the plan

        Returns: (plan_text, code_text)
        """
        plan = self._run_planner(planning_prompt, max_tokens)
        code = self._run_coder(coding_prompt + f"\n\nPlan to follow:\n{plan}", max_tokens)
        return plan, code

    @property
    def available_models(self) -> dict[str, str]:
        """Returns names of loaded models."""
        result = {}
        if self.planner:
            result["planner"] = self.planner.model_name
        if self.coder:
            result["coder"] = self.coder.model_name
        return result

    # ── Private ────────────────────────────────────────────────────────────────

    def _pick_loader(self, role: ModelRole) -> "ModelLoader":
        """Pick the best available loader for the given role, and unload the other to save RAM strictly (<50%)."""
        if role == ModelRole.CODER:
            # We want CODER active. Unload PLANNER if it has an active llm
            if self.planner and getattr(self.planner, 'llm', None) is not None:
                self.planner.unload()
            
            # Fallback logic if coder didn't exist strictly
            loader = self.coder or self.planner
        else:
            # We want PLANNER active. Unload CODER if it has an active llm
            if self.coder and getattr(self.coder, 'llm', None) is not None:
                self.coder.unload()
            
            # Fallback logic
            loader = self.planner or self.coder
            
        # Guarantee it's loaded before returning
        if loader and getattr(loader, 'llm', None) is None:
            loader.load()
            
        return loader

    def _run_planner(self, prompt: str, max_tokens: int) -> str:
        loader = self.planner or self.coder
        return loader.generate(prompt, max_tokens=max_tokens, temperature=0.3)

    def _run_coder(self, prompt: str, max_tokens: int) -> str:
        loader = self.coder or self.planner
        return loader.generate(prompt, max_tokens=max_tokens, temperature=0.1)
