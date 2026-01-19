"""Base compaction interface."""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from salesbench.context.buffers import Message


class Compactor(ABC):
    """Abstract base class for context compaction strategies."""

    @abstractmethod
    def compact(self, messages: List["Message"]) -> List["Message"]:
        """Compact a list of messages.

        Args:
            messages: Messages to compact.

        Returns:
            Compacted messages (fewer messages, preserving key info).
        """
        pass

    @abstractmethod
    def should_compact(self, messages: List["Message"], max_tokens: int) -> bool:
        """Check if compaction is needed.

        Args:
            messages: Current messages.
            max_tokens: Token limit.

        Returns:
            True if compaction should be performed.
        """
        pass
