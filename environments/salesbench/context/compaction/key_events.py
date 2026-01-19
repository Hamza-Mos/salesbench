"""Key events-based compaction."""

from typing import List, Set
from salesbench.context.compaction.base import Compactor
from salesbench.context.buffers import Message


# Event types that are always preserved
KEY_EVENT_TYPES = {
    "plan_accepted",
    "plan_rejected",
    "call_ended",
    "dnc_violation",
    "deal_closed",
    "appointment_scheduled",
}

# Keywords that indicate important messages
KEY_KEYWORDS = {
    "accepted",
    "rejected",
    "closed",
    "scheduled",
    "violation",
    "error",
    "success",
    "agreed",
    "confirmed",
}


class KeyEventsCompactor(Compactor):
    """Compacts messages while preserving key events.

    Key events (acceptances, rejections, closures) are always kept.
    Other messages are summarized or dropped based on age.
    """

    def __init__(
        self,
        key_event_types: Set[str] = None,
        key_keywords: Set[str] = None,
        keep_recent: int = 5,
        min_priority: int = 5,
    ):
        """Initialize the compactor.

        Args:
            key_event_types: Event types to preserve.
            key_keywords: Keywords that indicate important messages.
            keep_recent: Recent messages to always keep.
            min_priority: Minimum priority to preserve.
        """
        self.key_event_types = key_event_types or KEY_EVENT_TYPES
        self.key_keywords = key_keywords or KEY_KEYWORDS
        self.keep_recent = keep_recent
        self.min_priority = min_priority

    def compact(self, messages: List[Message]) -> List[Message]:
        """Compact while preserving key events."""
        if len(messages) <= self.keep_recent:
            return messages

        result = []
        recent_start = len(messages) - self.keep_recent

        for i, msg in enumerate(messages):
            # Always keep recent messages
            if i >= recent_start:
                result.append(msg)
                continue

            # Keep messages with high priority
            if msg.priority >= self.min_priority:
                result.append(msg)
                continue

            # Keep key events
            if self._is_key_event(msg):
                result.append(msg)
                continue

            # Keep messages with key keywords
            if self._has_key_keywords(msg):
                result.append(msg)
                continue

        # If we removed too many, add a summary
        removed_count = len(messages) - len(result)
        if removed_count > 0:
            summary = Message(
                role="system",
                content=f"[{removed_count} routine messages omitted]",
                priority=1,
            )
            result.insert(0, summary)

        return result

    def should_compact(self, messages: List[Message], max_tokens: int) -> bool:
        """Check if compaction is needed."""
        total_tokens = sum(m.token_estimate() for m in messages)
        return total_tokens > max_tokens * 0.75

    def _is_key_event(self, message: Message) -> bool:
        """Check if message represents a key event."""
        event_type = message.metadata.get("event_type", "")
        return event_type in self.key_event_types

    def _has_key_keywords(self, message: Message) -> bool:
        """Check if message contains key keywords."""
        content_lower = message.content.lower()
        return any(kw in content_lower for kw in self.key_keywords)


class ToolResultCompactor(Compactor):
    """Compacts tool results while preserving important ones.

    Tool results are often verbose. This compactor:
    - Keeps results from important tools (calling, crm.log_call)
    - Summarizes or drops routine tool results (crm.search_leads)
    """

    # Tools whose results are always kept
    IMPORTANT_TOOLS = {
        "calling.propose_plan",
        "calling.start_call",
        "calling.end_call",
        "crm.log_call",
        "calendar.schedule_call",
    }

    # Tools whose results can be summarized
    SUMMARIZABLE_TOOLS = {
        "crm.search_leads",
        "crm.get_lead",
        "products.list_plans",
        "products.get_plan",
        "products.quote_premium",
        "calendar.get_availability",
    }

    def __init__(self, keep_recent: int = 3):
        self.keep_recent = keep_recent

    def compact(self, messages: List[Message]) -> List[Message]:
        """Compact tool result messages."""
        result = []
        tool_results_seen = {}  # tool_name -> count

        for i, msg in enumerate(messages):
            # Keep non-tool messages
            if msg.role != "tool":
                result.append(msg)
                continue

            tool_name = msg.name or "unknown"

            # Always keep important tool results
            if tool_name in self.IMPORTANT_TOOLS:
                result.append(msg)
                continue

            # Track and potentially skip repeated tool results
            tool_results_seen[tool_name] = tool_results_seen.get(tool_name, 0) + 1

            # Keep first few of each type
            if tool_results_seen[tool_name] <= self.keep_recent:
                result.append(msg)
            # Otherwise skip (will be summarized)

        return result

    def should_compact(self, messages: List[Message], max_tokens: int) -> bool:
        """Check based on tool message ratio."""
        tool_messages = sum(1 for m in messages if m.role == "tool")
        return tool_messages > len(messages) * 0.5
