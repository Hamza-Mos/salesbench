"""Message buffers for context management.

Provides different buffering strategies for managing conversation history.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Optional
import heapq


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None  # Tool name for tool messages
    tool_call_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    timestamp: int = 0  # Monotonic counter

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
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


class SlidingWindowBuffer(MessageBuffer):
    """Fixed-size sliding window buffer.

    Keeps the N most recent messages.
    """

    def __init__(self, max_messages: int = 50, max_tokens: Optional[int] = None):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._messages: deque[Message] = deque(maxlen=max_messages)
        self._counter = 0

    def add(self, message: Message) -> None:
        message.timestamp = self._counter
        self._counter += 1
        self._messages.append(message)

        # Trim by tokens if needed
        if self.max_tokens:
            while self.token_count() > self.max_tokens and len(self._messages) > 1:
                self._messages.popleft()

    def get_messages(self) -> List[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._counter = 0

    def token_count(self) -> int:
        return sum(m.token_estimate() for m in self._messages)


class PriorityBuffer(MessageBuffer):
    """Priority-based buffer that keeps high-priority messages.

    When the buffer is full, removes lowest-priority messages first.
    """

    def __init__(
        self,
        max_messages: int = 50,
        max_tokens: Optional[int] = None,
        min_priority_to_keep: int = 0,
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.min_priority_to_keep = min_priority_to_keep
        self._messages: List[Message] = []
        self._counter = 0

    def add(self, message: Message) -> None:
        message.timestamp = self._counter
        self._counter += 1
        self._messages.append(message)

        # Trim by count
        while len(self._messages) > self.max_messages:
            self._remove_lowest_priority()

        # Trim by tokens
        if self.max_tokens:
            while self.token_count() > self.max_tokens and len(self._messages) > 1:
                self._remove_lowest_priority()

    def _remove_lowest_priority(self) -> None:
        """Remove the lowest priority, oldest message."""
        if not self._messages:
            return

        # Find lowest priority messages
        min_priority = min(m.priority for m in self._messages)

        # Among those, find the oldest
        candidates = [m for m in self._messages if m.priority == min_priority]
        oldest = min(candidates, key=lambda m: m.timestamp)

        self._messages.remove(oldest)

    def get_messages(self) -> List[Message]:
        # Return in timestamp order
        return sorted(self._messages, key=lambda m: m.timestamp)

    def clear(self) -> None:
        self._messages.clear()
        self._counter = 0

    def token_count(self) -> int:
        return sum(m.token_estimate() for m in self._messages)

    def set_priority(self, index: int, priority: int) -> None:
        """Update priority of a message by index."""
        if 0 <= index < len(self._messages):
            self._messages[index].priority = priority


class KeyEventBuffer(MessageBuffer):
    """Buffer that preserves key events and summarizes the rest.

    Key events are marked with high priority and always kept.
    Other messages may be summarized when the buffer is compacted.
    """

    def __init__(
        self,
        max_messages: int = 100,
        max_tokens: int = 4000,
        key_event_priority: int = 10,
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.key_event_priority = key_event_priority
        self._messages: List[Message] = []
        self._summaries: List[Message] = []  # Compacted summaries
        self._counter = 0

    def add(self, message: Message) -> None:
        message.timestamp = self._counter
        self._counter += 1
        self._messages.append(message)

    def add_key_event(self, content: str, role: str = "system") -> None:
        """Add a key event that should always be preserved."""
        message = Message(
            role=role,
            content=content,
            priority=self.key_event_priority,
            timestamp=self._counter,
        )
        self._counter += 1
        self._messages.append(message)

    def add_summary(self, summary: str) -> None:
        """Add a summary of compacted messages."""
        self._summaries.append(Message(
            role="system",
            content=f"[Summary of previous conversation]: {summary}",
            priority=self.key_event_priority - 1,
            timestamp=self._counter,
        ))
        self._counter += 1

    def get_messages(self) -> List[Message]:
        all_messages = self._summaries + self._messages
        return sorted(all_messages, key=lambda m: m.timestamp)

    def clear(self) -> None:
        self._messages.clear()
        self._summaries.clear()
        self._counter = 0

    def token_count(self) -> int:
        return sum(m.token_estimate() for m in self.get_messages())

    def compact(self, summarizer=None) -> None:
        """Compact old messages into a summary.

        Args:
            summarizer: Optional callable that takes messages and returns summary.
        """
        if len(self._messages) <= self.max_messages // 2:
            return

        # Separate key events from regular messages
        key_events = [m for m in self._messages if m.priority >= self.key_event_priority]
        regular = [m for m in self._messages if m.priority < self.key_event_priority]

        # Keep recent regular messages
        keep_count = self.max_messages // 4
        to_compact = regular[:-keep_count] if len(regular) > keep_count else []
        to_keep = regular[-keep_count:] if len(regular) > keep_count else regular

        if to_compact:
            if summarizer:
                summary = summarizer(to_compact)
            else:
                # Default: just note how many messages were compacted
                summary = f"[{len(to_compact)} earlier messages about lead interactions]"

            self.add_summary(summary)
            self._messages = key_events + to_keep
