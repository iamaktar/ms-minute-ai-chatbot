"""Conversation memory manager with a configurable sliding window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from google.genai import types

Role = Literal["user", "model"]


@dataclass(frozen=True, slots=True)
class Message:
    """Represents a single turn in the conversation."""

    role: Role
    text: str


class ConversationMemory:
    """Stores multi-turn chat history with a fixed-size sliding window.

    Ensures the retained history always starts with a 'user' message so that
    turn-taking constraints of the Gemini API are preserved after trimming.
    """

    def __init__(self, max_messages: int = 20) -> None:
        """Initialize the conversation memory.

        Args:
            max_messages: Maximum number of messages to retain in memory.

        Raises:
            ValueError: If max_messages is less than 2.
        """
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2.")
        self._max_messages: int = max_messages
        self._messages: list[Message] = []

    @property
    def max_messages(self) -> int:
        """Return the configured maximum message count."""
        return self._max_messages

    @property
    def messages(self) -> list[Message]:
        """Return a shallow copy of the current message history."""
        return list(self._messages)

    def add_user_message(self, text: str) -> None:
        """Append a user message and enforce the sliding window."""
        self._messages.append(Message(role="user", text=text))
        self._trim()

    def add_model_message(self, text: str) -> None:
        """Append a model response and enforce the sliding window."""
        self._messages.append(Message(role="model", text=text))
        self._trim()

    def pop_last_message(self) -> Message | None:
        """Remove and return the most recent message, if any."""
        if not self._messages:
            return None
        return self._messages.pop()

    def clear(self) -> None:
        """Erase all stored conversation history."""
        self._messages.clear()

    def to_genai_contents(self) -> list[types.Content]:
        """Convert stored messages into Google GenAI Content objects."""
        return [
            types.Content(
                role=msg.role,
                parts=[types.Part.from_text(text=msg.text)],
            )
            for msg in self._messages
        ]

    def format_history(self) -> str:
        """Return a human-readable string representation of the history."""
        if not self._messages:
            return "No messages in current session."
        lines: list[str] = []
        for idx, msg in enumerate(self._messages, start=1):
            label = "You" if msg.role == "user" else "Ms. Minute"
            lines.append(f"[{idx}] {label}:\n{msg.text}")
        return "\n\n".join(lines)

    def _trim(self) -> None:
        """Enforce the sliding window and ensure history begins with a user turn."""
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]
        while self._messages and self._messages[0].role != "user":
            self._messages.pop(0)