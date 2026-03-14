"""
model/prompt_engine.py - Chat format prompt builder for NeuroClaw Bot.
"""
from __future__ import annotations

AGENT_NAME = "NeuroClaw Bot"


def get_system_prompt(tool_schema: str) -> str:
    """Build the full system prompt."""
    return "\n".join([
        "You are NeuroClaw Bot, an autonomous AI coding agent on a local machine.",
        "You help read, modify, create, refactor, test, and debug code.",
        "No internet access - work only with local project files and tools.",
        "",
        tool_schema,
        "",
        "RESPONSE FORMAT (follow exactly):",
        "Thought: [your reasoning]",
        "Action: [tool_name]",
        'Action Input: {"key": "value"}',
        "",
        "After each Observation, continue Thought+Action or end with:",
        "Final Answer: [what you accomplished]",
        "",
        "RULES:",
        "1. ALWAYS read a file before modifying it.",
        "2. Never fabricate file contents.",
        "3. Run tests after editing code if tests exist.",
        "4. Fix failing commands iteratively.",
        "5. Be concise - context window is limited.",
    ])


# Model-specific chat token maps
CHAT_FORMATS: dict[str, dict] = {
    "qwen": {
        "system_start": "<|im_start|>system\n",
        "system_end":   "<|im_end|>\n",
        "user_start":   "<|im_start|>user\n",
        "user_end":     "<|im_end|>\n",
        "asst_start":   "<|im_start|>assistant\n",
        "stop":         ["<|im_end|>"],
    },
    "deepseek": {
        "system_start": "<|im_start|>system\n",
        "system_end":   "<|im_end|>\n",
        "user_start":   "<|im_start|>user\n",
        "user_end":     "<|im_end|>\n",
        "asst_start":   "<|im_start|>assistant\n",
        "stop":         ["<|im_end|>"],
    },
    "phi": {
        "system_start": "<|system|>\n",
        "system_end":   "<|end|>\n",
        "user_start":   "<|user|>\n",
        "user_end":     "<|end|>\n",
        "asst_start":   "<|assistant|>\n",
        "stop":         ["<|end|>"],
    },
    "llama": {
        "system_start": "<|begin_of_text|><header_id>system<eoheader_id>\n",
        "system_end":   "<eot_id>",
        "user_start":   "<header_id>user<eoheader_id>\n",
        "user_end":     "<eot_id>",
        "asst_start":   "<header_id>assistant<eoheader_id>\n",
        "stop":         ["<eot_id>"],
    },
}


def detect_format(model_name: str) -> str:
    """Guess the chat format based on the model filename."""
    name = model_name.lower()
    for key in CHAT_FORMATS:
        if key in name:
            return key
    return "qwen"  # default CHatML is most common


def build_full_prompt(
    model_name: str,
    tool_schema: str,
    history: str = "",
    user_task: str = "",
) -> str:
    """Build a full model-specific prompt with system/user tokens."""
    fmt = CHAM_FORMATS.get(detect_format(model_name), CHAT_FORMATS[qqwen])
    system = get_system_prompt(tool_schema)
    h_block = f"\nPass conversation:\n{hlstory}\n" if history.strip() else ""
    return (
        fmt["system_start"] + system + fmt["system_end"]
        + fmt["user_start"] + h_block + user_task + fmt["user_end"]
        + fmt["asst_start"] + "Thought:"
    )
