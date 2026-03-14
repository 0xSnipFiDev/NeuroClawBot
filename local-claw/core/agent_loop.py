"""
core/agent_loop.py — The ReAct (Reasoning + Acting) agent loop.

Implements the iterative loop:
  Goal → Thought → Action → Observation → Thought → ... → Final Answer

Inspired by:
- Claude Code's autonomous task execution with tool use
- OpenClaw's Pi agent runtime with tool streaming
"""
import re
import json
from model.loader import ModelLoader
from core.memory import ConversationMemory
from tools.registry import ToolRegistry

# Matches: Action: tool_name
ACTION_PATTERN = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
# Matches: Action Input: { ... } (handles both single-line and multiline JSON)
ACTION_INPUT_PATTERN = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL)
# Matches: Final Answer: ...
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


class AgentLoop:
    """
    Drives the ReAct agent loop.

    Each iteration:
    1. Builds a prompt (system + history + task + previous observations)
    2. Calls the LLM to get Thought + Action + Action Input
    3. Parses the tool call
    4. Executes the tool
    5. Appends the Observation and loops

    Exits on:
    - "Final Answer:" in model output
    - Max iterations reached
    """

    def __init__(
        self,
        loader: ModelLoader,
        tools: ToolRegistry,
        memory: ConversationMemory,
        max_iterations: int = 12,
        max_tokens_per_step: int = 512,
    ):
        self.loader = loader
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
        self.max_tokens_per_step = max_tokens_per_step

    def run(self, task: str) -> str:
        """
        Execute a user task and return the final answer string.

        Args:
            task: The user's instruction, possibly pre-enriched with RAG context.

        Returns:
            The agent's final answer string.
        """
        self.memory.add("user", task)

        # Build initial prompt
        system_prompt = self._build_system_prompt()
        history = self.memory.get_recent(max_tokens=1200)
        prompt = self._build_prompt(system_prompt, history, task)

        print(f"\n{'='*60}")
        print(f"[Agent] Starting task: {task[:120]}{'...' if len(task) > 120 else ''}")
        print(f"{'='*60}\n")

        for iteration in range(1, self.max_iterations + 1):
            print(f"─── Iteration {iteration}/{self.max_iterations} ───")

            # Generate next step from the model
            raw_output = self.loader.generate(
                prompt,
                max_tokens=self.max_tokens_per_step,
                stop=["Observation:", "\nUser:", "Human:", "<|im_end|>"],
            )
            print(f"[Model]\n{raw_output}\n")

            # ── Check for Final Answer ──────────────────────
            fa_match = FINAL_ANSWER_PATTERN.search(raw_output)
            if fa_match:
                answer = fa_match.group(1).strip()
                self.memory.add("assistant", answer)
                print(f"\n✅ [Agent] Task complete.\n")
                return answer

            # ── Parse Action + Action Input ─────────────────
            action_match = ACTION_PATTERN.search(raw_output)
            if not action_match:
                # Model didn't follow the format — nudge it
                nudge = (
                    "\nObservation: [Format error] You must respond with "
                    "'Action: tool_name' and 'Action Input: {\"key\": \"value\"}', "
                    "or 'Final Answer: your_answer'.\n"
                )
                prompt += raw_output + nudge
                continue

            tool_name = action_match.group(1).strip()
            input_match = ACTION_INPUT_PATTERN.search(raw_output)
            tool_args: dict = {}

            if input_match:
                try:
                    tool_args = json.loads(input_match.group(1))
                except json.JSONDecodeError as e:
                    # Try to fix common issues (single quotes, trailing commas)
                    fixed = input_match.group(1).replace("'", '"')
                    try:
                        tool_args = json.loads(fixed)
                    except Exception:
                        observation = f"[ERROR] Could not parse Action Input as JSON: {e}"
                        prompt += raw_output + f"\nObservation: {observation}\n"
                        continue

            # ── Execute Tool ────────────────────────────────
            print(f"[Tool] {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})")
            observation = self.tools.call(tool_name, tool_args)

            # Truncate long observations to protect context
            if len(observation) > 3000:
                observation = (
                    observation[:3000]
                    + "\n...[TRUNCATED — use search_code() for specific queries]"
                )
            print(f"[Observation] {observation[:300]}{'...' if len(observation) > 300 else ''}\n")

            # Append to prompt for next iteration
            prompt += raw_output + f"\nObservation: {observation}\nThought:"

        # Max iterations reached
        partial_answer = (
            f"[Agent] Reached maximum iterations ({self.max_iterations}). "
            "The task may be partially complete. Please review the changes made "
            "and continue with a more specific instruction."
        )
        self.memory.add("assistant", partial_answer)
        return partial_answer

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return f"""\
You are NeuroClaw_Bot, an autonomous AI coding agent running entirely on a local machine.
You help users read, modify, create, refactor, and debug code.

{self.tools.schema_description}

RESPONSE FORMAT (strict):
Thought: [reasoning]
Action: [tool_name]
Action Input: {{"key": "value"}}

When done:
Final Answer: [summary of what you accomplished]

RULES:
- Always read files before modifying them.
- Make targeted, minimal changes. Never hallucinate file contents.
- Run tests if they exist after making changes.
- Think step by step. Be concise.
"""

    def _build_prompt(self, system_prompt: str, history: str, task: str) -> str:
        history_block = f"\nPrevious conversation:\n{history}\n" if history.strip() else ""
        return (
            system_prompt
            + history_block
            + f"\nUser Task: {task}\n\nThought:"
        )
