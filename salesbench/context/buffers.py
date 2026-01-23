"""Message buffers for context management.

Provides different buffering strategies for managing conversation history.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None  # Tool name for tool messages
    tool_call_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    priority: int = 0  # Higher = more important (used by LLMCompactBuffer for key events)
    timestamp: int = 0  # Monotonic counter

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        # Include gemini_content for thought_signature preservation (Gemini 3 requirement)
        if "gemini_content" in self.metadata:
            result["gemini_content"] = self.metadata["gemini_content"]
        return result

    def token_estimate(self) -> int:
        """Rough token count estimate."""
        # Approximate: 4 chars per token
        return len(self.content) // 4 + 10  # +10 for role/structure overhead


class MessageBuffer(ABC):
    """Abstract base class for message buffers."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Add a message to the buffer."""
        pass

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """Get all messages in order."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages."""
        pass

    @abstractmethod
    def token_count(self) -> int:
        """Estimate total token count."""
        pass

    def to_api_messages(self) -> List[dict[str, Any]]:
        """Convert to API format."""
        return [m.to_dict() for m in self.get_messages()]


class LLMCompactBuffer(MessageBuffer):
    """Buffer with LLM-based compaction.

    When context exceeds threshold, older messages are sent to an LLM
    to produce a structured memory summary. Recent messages are kept verbatim.
    """

    def __init__(
        self,
        max_tokens: int = 100_000,
        keep_recent: int = 10,
        compaction_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        """Initialize the buffer.

        Args:
            max_tokens: Maximum tokens before triggering compaction.
            keep_recent: Number of recent messages to keep verbatim.
            compaction_fn: Async function that takes message text and returns summary.
        """
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self._messages: List[Message] = []
        self._compacted_memory: Optional[str] = None
        self._compaction_fn = compaction_fn
        self._counter = 0

    def add(self, message: Message) -> None:
        """Add a message to the buffer."""
        message.timestamp = self._counter
        self._counter += 1
        self._messages.append(message)

    def add_key_event(self, content: str, role: str = "system") -> None:
        """Add a key event message.

        Key events are marked with higher priority for logging/tracking purposes.
        With LLM compaction, all older messages get summarized together.

        Args:
            content: The event content.
            role: Message role (default: system).
        """
        self._messages.append(
            Message(
                role=role,
                content=content,
                priority=10,  # Mark as key event for reference
                timestamp=self._counter,
            )
        )
        self._counter += 1

    async def compact_if_needed(self, trigger_tokens: int) -> bool:
        """Compact using LLM when exceeding trigger.

        Args:
            trigger_tokens: Token count threshold to trigger compaction.

        Returns:
            True if compaction was performed, False otherwise.
        """
        current_tokens = self.token_count()
        if current_tokens <= trigger_tokens:
            return False

        if len(self._messages) <= self.keep_recent:
            return False

        before_count = len(self._messages)
        older = self._messages[: -self.keep_recent]
        recent = self._messages[-self.keep_recent :]

        logger.info(
            f"[Seller Context] COMPACTING: {current_tokens:,} > {trigger_tokens:,} tokens "
            f"({before_count} messages)"
        )

        if self._compaction_fn:
            older_text = self._format_messages(older)
            self._compacted_memory = await self._compaction_fn(older_text)

        self._messages = recent
        after_tokens = self.token_count()

        logger.info(
            f"[Seller Context] COMPACTED via LLM: {after_tokens:,} tokens "
            f"({len(recent)} messages kept, {len(older)} messages summarized)"
        )
        return True

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for compaction prompt.

        Args:
            messages: Messages to format.

        Returns:
            Formatted string for the compaction prompt.
        """
        lines = []
        for msg in messages:
            lines.append(f"[{msg.role}]: {msg.content}")
        return "\n".join(lines)

    def get_messages(self) -> List[Message]:
        """Get all messages in order."""
        return list(self._messages)

    def get_memory_prefix(self) -> Optional[str]:
        """Get compacted memory to inject before messages.

        Returns:
            The compacted memory summary, or None if no compaction occurred.
        """
        return self._compacted_memory

    def token_count(self) -> int:
        """Estimate total token count."""
        base = sum(m.token_estimate() for m in self._messages)
        if self._compacted_memory:
            base += len(self._compacted_memory) // 4 + 10
        return base

    def clear(self) -> None:
        """Clear all messages and compacted memory."""
        self._messages.clear()
        self._compacted_memory = None
        self._counter = 0
