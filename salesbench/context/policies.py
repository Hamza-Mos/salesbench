"""Context management policies.

Policies define how context is managed, including when to compact,
what to preserve, and token budget allocation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from salesbench.context.buffers import MessageBuffer
    from salesbench.context.compaction.base import Compactor


@dataclass
class ContextBudget:
    """Token budget allocation for context."""

    total_tokens: int = 8000  # Total available tokens
    system_tokens: int = 1000  # Reserved for system prompt
    response_tokens: int = 1000  # Reserved for response
    history_tokens: int = 6000  # Available for conversation history

    # Sub-allocations within history
    key_events_ratio: float = 0.3  # 30% for key events
    recent_messages_ratio: float = 0.5  # 50% for recent messages
    summaries_ratio: float = 0.2  # 20% for summaries

    def validate(self) -> bool:
        """Check budget consistency."""
        return (
            self.system_tokens + self.response_tokens + self.history_tokens <= self.total_tokens
            and abs(self.key_events_ratio + self.recent_messages_ratio + self.summaries_ratio - 1.0)
            < 0.01
        )


class ContextPolicy(ABC):
    """Abstract base class for context policies."""

    @abstractmethod
    def should_compact(self, buffer: "MessageBuffer") -> bool:
        """Determine if compaction is needed."""
        pass

    @abstractmethod
    def get_compactor(self) -> "Compactor":
        """Get the compactor to use."""
        pass

    @abstractmethod
    def get_budget(self) -> ContextBudget:
        """Get the token budget."""
        pass

    @abstractmethod
    def prepare_context(
        self,
        buffer: "MessageBuffer",
        system_prompt: str,
    ) -> List[dict]:
        """Prepare context for API call.

        Args:
            buffer: Message buffer.
            system_prompt: System prompt to include.

        Returns:
            List of messages ready for API call.
        """
        pass


class DefaultContextPolicy(ContextPolicy):
    """Default context management policy.

    Uses sliding window with simple summary compaction.
    """

    def __init__(
        self,
        budget: Optional[ContextBudget] = None,
        compact_threshold: float = 0.8,
    ):
        self.budget = budget or ContextBudget()
        self.compact_threshold = compact_threshold
        self._compactor = None

    def should_compact(self, buffer: "MessageBuffer") -> bool:
        current_tokens = buffer.token_count()
        max_tokens = self.budget.history_tokens
        return current_tokens > max_tokens * self.compact_threshold

    def get_compactor(self) -> "Compactor":
        if self._compactor is None:
            from salesbench.context.compaction.simple_summary import SimpleSummaryCompactor

            self._compactor = SimpleSummaryCompactor(keep_recent=10)
        return self._compactor

    def get_budget(self) -> ContextBudget:
        return self.budget

    def prepare_context(
        self,
        buffer: "MessageBuffer",
        system_prompt: str,
    ) -> List[dict]:
        # Check if compaction needed
        if self.should_compact(buffer):
            messages = buffer.get_messages()
            compacted = self.get_compactor().compact(messages)
            buffer.clear()
            for msg in compacted:
                buffer.add(msg)

        # Build context
        result = [{"role": "system", "content": system_prompt}]
        result.extend(buffer.to_api_messages())

        return result


class AggressiveCompactionPolicy(ContextPolicy):
    """Aggressive compaction for limited context windows.

    Compacts early and often, keeping only essentials.
    """

    def __init__(
        self,
        budget: Optional[ContextBudget] = None,
        keep_recent: int = 5,
    ):
        self.budget = budget or ContextBudget(
            total_tokens=4000,
            system_tokens=500,
            response_tokens=500,
            history_tokens=3000,
        )
        self.keep_recent = keep_recent
        self._compactor = None

    def should_compact(self, buffer: "MessageBuffer") -> bool:
        # Compact at 60% capacity
        return buffer.token_count() > self.budget.history_tokens * 0.6

    def get_compactor(self) -> "Compactor":
        if self._compactor is None:
            from salesbench.context.compaction.key_events import KeyEventsCompactor

            self._compactor = KeyEventsCompactor(keep_recent=self.keep_recent)
        return self._compactor

    def get_budget(self) -> ContextBudget:
        return self.budget

    def prepare_context(
        self,
        buffer: "MessageBuffer",
        system_prompt: str,
    ) -> List[dict]:
        messages = buffer.get_messages()

        # Always compact
        compacted = self.get_compactor().compact(messages)

        result = [{"role": "system", "content": system_prompt}]
        for msg in compacted:
            result.append(msg.to_dict())

        return result


class PreserveRecentPolicy(ContextPolicy):
    """Policy that prioritizes recent messages.

    Keeps all recent messages and aggressively summarizes older ones.
    """

    def __init__(
        self,
        budget: Optional[ContextBudget] = None,
        recent_turns: int = 10,
    ):
        self.budget = budget or ContextBudget()
        self.recent_turns = recent_turns
        self._compactor = None

    def should_compact(self, buffer: "MessageBuffer") -> bool:
        messages = buffer.get_messages()
        return len(messages) > self.recent_turns * 2

    def get_compactor(self) -> "Compactor":
        if self._compactor is None:
            from salesbench.context.compaction.simple_summary import SimpleSummaryCompactor

            self._compactor = SimpleSummaryCompactor(keep_recent=self.recent_turns)
        return self._compactor

    def get_budget(self) -> ContextBudget:
        return self.budget

    def prepare_context(
        self,
        buffer: "MessageBuffer",
        system_prompt: str,
    ) -> List[dict]:
        messages = buffer.get_messages()

        # Keep recent, summarize rest
        if len(messages) > self.recent_turns:
            compacted = self.get_compactor().compact(messages)
        else:
            compacted = messages

        result = [{"role": "system", "content": system_prompt}]
        for msg in compacted:
            result.append(msg.to_dict())

        return result


class CallContextPolicy(ContextPolicy):
    """Specialized policy for call contexts.

    Preserves:
    - Lead information (always)
    - All offers presented in this call
    - Buyer responses
    - Recent conversation
    """

    def __init__(self, budget: Optional[ContextBudget] = None):
        self.budget = budget or ContextBudget(
            total_tokens=4000,
            system_tokens=800,
            response_tokens=500,
            history_tokens=2700,
        )

    def should_compact(self, buffer: "MessageBuffer") -> bool:
        # Only compact if really full
        return buffer.token_count() > self.budget.history_tokens * 0.9

    def get_compactor(self) -> "Compactor":
        from salesbench.context.compaction.key_events import KeyEventsCompactor

        return KeyEventsCompactor(
            key_keywords={"offer", "accept", "reject", "premium", "coverage"},
            keep_recent=3,
        )

    def get_budget(self) -> ContextBudget:
        return self.budget

    def prepare_context(
        self,
        buffer: "MessageBuffer",
        system_prompt: str,
    ) -> List[dict]:
        messages = buffer.get_messages()

        # Light compaction if needed
        if self.should_compact(buffer):
            messages = self.get_compactor().compact(messages)

        result = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            result.append(msg.to_dict())

        return result
