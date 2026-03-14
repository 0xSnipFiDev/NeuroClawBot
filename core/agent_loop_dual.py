"""
core/agent_loop_dual.py — Dual-model ReAct agent loop for NeuroClaw Bot.

This extends the base AgentLoop to use the ModelRouter:
- Phi-3-mini (planner) for reasoning / planning steps
- Qwen2.5-Coder-3B (coder) for code writing / editing steps

Two-phase mode (--two-phase flag):
  Step 1: Phi-3-mini creates a structured plan
  Step 2: Qwen-Coder executes the plan tool-by-tool
"""

# ── Bootstrap: add project root to sys.path so imports work from any CWD ──────
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # local-claw/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ──────────────────────────────────────────────────────────────────────────────

import re
import json
from model.model_router import ModelRouter, ModelRole
from core.memory import ConversationMemory
from tools.registry import ToolRegistry

ACTION_PATTERN = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
ACTION_INPUT_PATTERN = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


class DualModelAgentLoop:
    """
    ReAct agent loop powered by two local models:
      🧠 Phi-3-mini  → used when role == PLANNER
      ⚙️  Qwen-Coder  → used when role == CODER or for code edits

    The router automatically selects the right model based on task keywords.
    Users can also force a role with `use coder` / `use planner` CLI commands.
    """

    def __init__(
        self,
        router: ModelRouter,
        tools: ToolRegistry,
        memory: ConversationMemory,
        max_iterations: int = 12,
        max_tokens_per_step: int = 512,
        two_phase: bool = False,
    ):
        self.router = router
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
        self.max_tokens_per_step = max_tokens_per_step
        self.two_phase = two_phase

    def run(self, task: str, force_role: ModelRole | None = None) -> str:
        """
        Execute a task and return the final answer.

        Args:
            task: User task string (may contain RAG context).
            force_role: Manually override which model to use.
        """
        self.memory.add("user", task)

        system_prompt = self._build_system_prompt()
        history = self.memory.get_recent(max_tokens=1200)
        prompt = self._build_prompt(system_prompt, history, task)

        print(f"\n{'='*60}")
        print(f"[Agent] Task: {task[:100]}{'...' if len(task) > 100 else ''}")

        # ── Optional two-phase mode ────────────────────────────────────────
        if self.two_phase and self.router.planner and self.router.coder:
            return self._run_two_phase(task, prompt, system_prompt)

        print(f"{'='*60}\n")

        for iteration in range(1, self.max_iterations + 1):
            print(f"─── Iteration {iteration}/{self.max_iterations} ───")

            # Pick which model to use for this iteration
            # After the first step: if the last action was a write/run → coder, else router decides
            raw = self.router.generate(
                prompt=prompt,
                task=task,
                max_tokens=self.max_tokens_per_step,
                temperature=0.1,
                stop=["Observation:", "\nUser:", "Human:", "<|end|>", "<|im_end|>"],
                force_role=force_role,
            )
            print(f"[{self.router.last_used_model}]\n{raw}\n")

            # ── Final Answer? ──────────────────────────────────────────────
            fa_match = FINAL_ANSWER_PATTERN.search(raw)
            if fa_match:
                answer = fa_match.group(1).strip()
                self.memory.add("assistant", answer)
                print(f"\n✅ [Agent] Done.\n")
                return answer

            # ── Parse tool call ────────────────────────────────────────────
            action_match = ACTION_PATTERN.search(raw)
            if not action_match:
                prompt += raw + (
                    "\nObservation: [Format error] Use:\n"
                    "Action: tool_name\nAction Input: {\"key\": \"value\"}\n"
                    "or: Final Answer: your_answer\nThought:"
                )
                continue

            tool_name = action_match.group(1).strip()
            input_match = ACTION_INPUT_PATTERN.search(raw)
            tool_args: dict = {}

            if input_match:
                json_str = input_match.group(1)
                try:
                    tool_args = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try fixing single quotes
                    try:
                        tool_args = json.loads(json_str.replace("'", '"'))
                    except Exception:
                        pass

            # After a write_file or run_terminal_command, next iteration should use coder
            if tool_name in ("write_file", "run_terminal_command") and not force_role:
                force_role = ModelRole.CODER

            # ── Execute tool ───────────────────────────────────────────────
            print(f"[Tool] {tool_name}({json.dumps(tool_args)[:180]})")
            observation = self.tools.call(tool_name, tool_args)
            if len(observation) > 3000:
                observation = observation[:3000] + "\n...[TRUNCATED]"

            print(f"[Obs] {observation[:250]}{'...' if len(observation) > 250 else ''}\n")
            prompt += raw + f"\nObservation: {observation}\nThought:"

        partial = (
            f"[Max iterations ({self.max_iterations}) reached. "
            "Review changes made and continue with a more specific task.]"
        )
        self.memory.add("assistant", partial)
        return partial

    # ── Two-phase execution ────────────────────────────────────────────────────

    def _run_two_phase(self, task: str, base_prompt: str, system_prompt: str) -> str:
        """
        Phase 1: Phi-3-mini plans the task step by step.
        Phase 2: Qwen-Coder executes each step.
        """
        print("[Two-phase mode] 🧠 Phase 1: Phi-3 planning...")
        plan_prompt = (
            system_prompt
            + f"\nUser Task: {task}\n\n"
            "First, list the exact steps needed to complete this task.\n"
            "Write a numbered plan. Be specific about which files to read and modify.\n\nPlan:\n"
        )
        plan = self.router.generate(
            prompt=plan_prompt,
            task=task,
            max_tokens=350,
            temperature=0.3,
            stop=["User:", "<|end|>", "<|im_end|>"],
            force_role=ModelRole.PLANNER,
        )
        print(f"[Phi-3 Plan]\n{plan}\n")

        print("[Two-phase mode] ⚙️  Phase 2: Qwen-Coder executing...")
        exec_prompt = (
            system_prompt
            + f"\nUser Task: {task}\n\n"
            f"Here is the plan to execute:\n{plan}\n\n"
            "Now execute the plan step by step using the available tools.\n\nThought:"
        )
        # Run Qwen-Coder loop
        return self._execute_loop(task, exec_prompt, force_role=ModelRole.CODER)

    def _execute_loop(self, task: str, prompt: str, force_role: ModelRole) -> str:
        """Inner execution loop for two-phase mode."""
        for iteration in range(1, self.max_iterations + 1):
            try:
                raw = self.router.generate(
                    prompt=prompt,
                    task=task,
                    max_tokens=self.max_tokens_per_step,
                    temperature=0.1,
                    stop=["Observation:", "\nUser:", "<|end|>", "<|im_end|>"],
                    force_role=force_role,
                )
            except ValueError as e:
                if "exceed context window" in str(e):
                    print("  ⚠ Context window exceeded. Clearing memory and retrying without RAG...")
                    self.memory.clear()
                    # Rebuild prompt without RAG context (assuming task might contain it)
                    # For now, we'll just rebuild the prompt with cleared memory.
                    system_prompt = self._build_system_prompt()
                    history = self.memory.get_recent(max_tokens=1200)
                    prompt = self._build_prompt(system_prompt, history, task)
                    raw = self.router.generate(
                        prompt=prompt,
                        task=task,
                        max_tokens=self.max_tokens_per_step,
                        temperature=0.1,
                        stop=["Observation:", "\nUser:", "<|end|>", "<|im_end|>"],
                        force_role=force_role,
                    )
                else:
                    raise

            print(f"[{self.router.last_used_model}]\n{raw}\n")

            fa_match = FINAL_ANSWER_PATTERN.search(raw)
            if fa_match:
                answer = fa_match.group(1).strip()
                self.memory.add("assistant", answer)
                return answer

            action_match = ACTION_PATTERN.search(raw)
            if not action_match:
                prompt += raw + "\nObservation: [Format error — use Action: / Action Input: / Final Answer:]\nThought:"
                continue

            tool_name = action_match.group(1).strip()
            input_match = ACTION_INPUT_PATTERN.search(raw)
            tool_args: dict = {}
            if input_match:
                try:
                    tool_args = json.loads(input_match.group(1))
                except Exception:
                    pass

            print(f"[Tool] {tool_name}({json.dumps(tool_args)[:180]})")
            observation = self.tools.call(tool_name, tool_args)
            if len(observation) > 3000:
                observation = observation[:3000] + "\n...[TRUNCATED]"
            print(f"[Obs] {observation[:200]}...\n")
            prompt += raw + f"\nObservation: {observation}\nThought:"

        return "[Max iterations reached]"

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return f"""\
You are NeuroClaw Bot, an autonomous AI coding agent running on a local machine.
You have access to TWO specialized models:
 - Phi-3-mini (planner): reasons, plans, explains, answers questions
 - Qwen2.5-Coder-3B (coder): writes, edits, refactors, and debugs code

{self.tools.schema_description}

RESPONSE FORMAT (strict):
Thought: [your reasoning]
Action: [tool_name]
Action Input: {{"key": "value"}}

When done:
Final Answer: [what you accomplished]

RULES:
- Always read a file before modifying it.
- Make targeted, minimal edits — never hallucinate content.
- Run tests after editing code if tests exist.
- Think step by step.
"""

    def _build_prompt(self, system: str, history: str, task: str) -> str:
        h = f"\nPast conversation:\n{history}\n" if history.strip() else ""
        return system + h + f"\nUser Task: {task}\n\nThought:"
