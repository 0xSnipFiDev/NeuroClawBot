"""
core/memory.py — Short-term conversation memory manager.
Keeps recent messages within a token budget to fit the model's context window.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Literal


Role = Literal["user", "assistant", "system", "observation"]


@dataclass
class Message:
    role: Role
    content: str


class ConversationMemory:
    """
    A sliding window of recent conversation messages.
    Estimates token count using a simple chars/4 heuristic.
    """

    def __init__(self, max_tokens: int = 2000, max_messages: int = 40):
        self.max_tokens = max_tokens
        self._messages: deque[Message] = deque(maxlen=max_messages)

    def add(self, role: Role, content: str):
        """Add a message to memory."""
        self._messages.append(Message(role=role, content=content))

    def get_recent(self, max_tokens: int | None = None) -> str:
        """
        Return recent conversation as a formatted string, fitting within the token budget.
        Oldest messages are dropped first if budget is exceeded.
        """
        budget = max_tokens or self.max_tokens
        lines = []
        token_count = 0

        for msg in reversed(self._messages):
            line = f"{msg.role.capitalize()}: {msg.content}"
            tokens = len(line) // 4  # rough estimate: 4 chars ≈ 1 token
            if token_count + tokens > budget:
                break
            lines.append(line)
            token_count += tokens

        return "\n".join(reversed(lines))

    def clear(self):
        """Reset memory (e.g., when starting a new task)."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
