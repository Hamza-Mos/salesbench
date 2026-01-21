"""Episode-wide context management for conversation history.

Manages conversation context for both seller (episode-wide) and buyer (per-lead) agents
with model-aware compression based on context window sizes.

Architecture:
- Seller: Episode-wide context - sees ALL conversations with ALL leads
- Buyer: Per-lead context - sees ONLY their own conversation with the seller

Context managers use ModelConfig to determine compression thresholds dynamically
based on each model's actual context window size.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

from salesbench.context.buffers import KeyEventBuffer, Message, SimpleCompactBuffer
from salesbench.core.types import PlanOffer
from salesbench.models import ModelConfig, get_model_config


@dataclass
class LeadConversation:
    """Conversation history for a single lead.

    Tracks all interactions between seller and a specific lead,
    including offers presented and buyer decisions.

    dialogue_only: Contains ONLY actual spoken text (what seller said, what buyer said)
    messages: Full history including system events (for seller view)
    """

    lead_id: str
    messages: list[Message] = field(default_factory=list)
    dialogue_only: list[tuple[str, str]] = field(
        default_factory=list
    )  # (role, text) - "seller"/"buyer"
    offers_presented: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    call_count: int = 0


@dataclass
class LeadSummary:
    """Minimal lead info that survives compaction."""

    lead_id: str
    name: str
    temperature: str
    income: int
    found_at_turn: int


@dataclass
class DecisionRecord:
    """Record of a buyer decision."""

    lead_id: str
    decision: str  # "accept_plan", "reject_plan", "end_call"
    turn: int


@dataclass
class AnchoredState:
    """Critical state that survives compaction - injected fresh each turn.

    This solves the problem of search results being lost during context compression.
    Instead of relying on message history, we maintain a separate state that is
    always injected at the start of the seller's view.
    """

    found_leads: dict[str, LeadSummary] = field(default_factory=dict)
    decisions: list[DecisionRecord] = field(default_factory=list)
    accepted_lead_ids: set[str] = field(default_factory=set)
    called_lead_ids: set[str] = field(default_factory=set)
    current_turn: int = 0
    # Track the currently active call
    active_call_lead_id: Optional[str] = None
    active_call_lead_name: Optional[str] = None
    # Track searched temperatures to avoid redundant searches
    searched_temperatures: set[str] = field(default_factory=set)

    def record_search(self, filters: dict) -> None:
        """Record that a search was performed with given filters."""
        if temp := filters.get("temperature"):
            self.searched_temperatures.add(temp)

    def record_search_results(self, leads: list[dict], turn: int) -> None:
        """Anchor leads found - these survive any compaction."""
        for lead in leads:
            lead_id = lead.get("lead_id")
            if lead_id and lead_id not in self.accepted_lead_ids:
                self.found_leads[lead_id] = LeadSummary(
                    lead_id=lead_id,
                    name=lead.get("name", "Unknown"),
                    temperature=lead.get("temperature", "unknown"),
                    income=lead.get("annual_income", 0),
                    found_at_turn=turn,
                )
            # Track the temperature that was searched
            if temp := lead.get("temperature"):
                self.searched_temperatures.add(temp)

    def record_call_started(self, lead_id: str, lead_name: str = "Unknown") -> None:
        """Record that a call was started with a lead."""
        self.called_lead_ids.add(lead_id)
        self.active_call_lead_id = lead_id
        self.active_call_lead_name = lead_name

    def record_call_ended(self, lead_id: str) -> None:
        """Record that a call ended."""
        if self.active_call_lead_id == lead_id:
            self.active_call_lead_id = None
            self.active_call_lead_name = None

    def record_decision(self, lead_id: str, decision: str, turn: int) -> None:
        """Record a buyer's decision."""
        self.decisions.append(DecisionRecord(lead_id, decision, turn))
        if decision == "accept_plan":
            self.accepted_lead_ids.add(lead_id)
            self.found_leads.pop(lead_id, None)
        # Call ends after any decision
        self.record_call_ended(lead_id)

    def get_uncalled_leads(self) -> list[LeadSummary]:
        """Get leads that haven't been called yet."""
        return [
            l
            for l in self.found_leads.values()
            if l.lead_id not in self.called_lead_ids and l.lead_id not in self.accepted_lead_ids
        ]

    def to_context_block(self) -> str:
        """Generate context block injected at start of every seller view.

        Shows facts clearly - no directive language. The model decides based on context.
        """
        lines = []

        # ACTIVE CALL STATUS - just the fact
        if self.active_call_lead_id:
            lines.append(f"ACTIVE CALL: {self.active_call_lead_name} ({self.active_call_lead_id})")
            lines.append("")

        # AVAILABLE LEADS (uncalled from found_leads)
        uncalled = self.get_uncalled_leads()
        if uncalled:
            lines.append(f"AVAILABLE LEADS ({len(uncalled)}):")
            for lead in uncalled:
                lines.append(
                    f"  {lead.lead_id}: {lead.name} ({lead.temperature}, ${lead.income:,}/yr)"
                )
            lines.append("")

        # SEARCH HISTORY - just show what was searched
        if self.searched_temperatures:
            lines.append(f"SEARCHED: {', '.join(sorted(self.searched_temperatures))}")

        # COMPLETED
        if self.accepted_lead_ids:
            lines.append(f"ACCEPTED: {len(self.accepted_lead_ids)} leads")

        # CALLED BUT NOT ACCEPTED
        rejected = self.called_lead_ids - self.accepted_lead_ids
        if rejected:
            lines.append(f"CALLED (not accepted): {len(rejected)} leads")

        return "\n".join(lines) if lines else ""

    def prune_old_leads(self, current_turn: int, max_age: int = 50) -> None:
        """Remove leads older than max_age turns that were never called."""
        to_remove = [
            lid
            for lid, lead in self.found_leads.items()
            if current_turn - lead.found_at_turn > max_age and lid not in self.called_lead_ids
        ]
        for lid in to_remove:
            del self.found_leads[lid]


class BaseContextManager(ABC):
    """Base class for model-aware context management."""

    def __init__(self, model_config: ModelConfig):
        """Initialize with model configuration.

        Args:
            model_config: Configuration for the model (context window, etc.)
        """
        self.model_config = model_config
        self._buffer = KeyEventBuffer(
            max_messages=200,
            max_tokens=model_config.available_context,
            key_event_priority=10,
        )

    @property
    def compression_trigger(self) -> int:
        """Token count at which to trigger compression."""
        return self.model_config.compression_trigger

    @abstractmethod
    def maybe_compress(self) -> None:
        """Compress if threshold exceeded."""
        pass

    def token_count(self) -> int:
        """Get current token count estimate."""
        return self._buffer.token_count()


class SellerContextManager(BaseContextManager):
    """Manages seller's episode-wide context.

    Seller sees ALL conversations with ALL leads.
    Uses SimpleCompactBuffer with observation masking (not summarization).
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        # Use SimpleCompactBuffer with observation masking
        self._buffer = SimpleCompactBuffer(
            max_tokens=model_config.available_context,
            keep_recent=20,
            key_event_priority=10,
        )
        self._lead_conversations: dict[str, LeadConversation] = {}

    @property
    def compression_trigger(self) -> int:
        """Token count at which to trigger compression."""
        return self.model_config.compression_trigger

    def maybe_compress(self) -> None:
        """Compress when exceeding model's context threshold using observation masking."""
        self._buffer.compact_if_needed(self.compression_trigger)

    def get_view(self) -> list[dict]:
        """Get seller's full episode history (compressed)."""
        self.maybe_compress()
        return self._buffer.to_api_messages()

    def add(self, message: Message) -> None:
        """Add a message to the buffer."""
        self._buffer.add(message)

    def add_key_event(self, content: str, role: str = "system") -> None:
        """Add a key event that should be preserved during compression."""
        self._buffer.add_key_event(content, role=role)

    def token_count(self) -> int:
        """Get current token count estimate."""
        return self._buffer.token_count()


class BuyerContextManager(BaseContextManager):
    """Manages buyer's per-lead context.

    Buyer sees ONLY their own conversation with seller.
    Uses model's actual context window instead of hardcoded 8K cap.
    Simpler compression - just keep recent dialogue.
    """

    def __init__(self, model_config: ModelConfig):
        # Use model's actual context window (no artificial 8K cap)
        super().__init__(model_config)
        self._dialogue: list[tuple[str, str]] = []  # (role, text)

    def maybe_compress(self) -> None:
        """Compress dialogue when exceeding threshold."""
        tokens = self._estimate_tokens()
        trigger = self.compression_trigger

        if tokens > trigger:
            # Keep last 4 turns minimum
            while len(self._dialogue) > 4 and self._estimate_tokens() > trigger:
                self._dialogue.pop(0)
            print(f"  [Buyer Context] Compressed to {len(self._dialogue)} turns")

    def _estimate_tokens(self) -> int:
        """Estimate token count for current dialogue."""
        return sum(len(text) // 4 + 15 for _, text in self._dialogue)

    def add_dialogue(self, role: str, text: str) -> None:
        """Add a dialogue turn."""
        self._dialogue.append((role, text))

    def get_view(self) -> str:
        """Get buyer's conversation history as formatted text."""
        self.maybe_compress()
        if not self._dialogue:
            return "This is the first interaction with this seller."
        lines = ["Conversation so far:"]
        for role, text in self._dialogue:
            speaker = "Seller" if role == "seller" else "You"
            lines.append(f'  {speaker}: "{text}"')
        return "\n".join(lines)

    def token_count(self) -> int:
        """Override to use dialogue-based estimate."""
        return self._estimate_tokens()


class EpisodeContext:
    """Manages conversation context for entire episode using modular managers.

    Provides two views:
    - get_seller_view(): Full episode history (compressed) for seller LLM
    - get_buyer_view(lead_id): Only this lead's conversation history

    Uses model-aware SellerContextManager and BuyerContextManager for intelligent
    compression based on each model's actual context window size.

    Example:
        # With model configs
        seller_cfg = get_model_config("gpt-5.2")  # 400K context
        buyer_cfg = get_model_config("gpt-4o-mini")  # 128K context

        context = EpisodeContext(
            seller_model_config=seller_cfg,
            buyer_model_config=buyer_cfg,
        )

        # Seller searches and calls leads
        context.record_seller_action("Searching for hot leads")
        context.record_tool_result("crm.search_leads", {"leads": [...]})
        context.record_call_start("lead_001")

        # When presenting offer, record it
        context.record_offer("lead_001", offer)
        context.record_buyer_decision("lead_001", "reject_plan", "Too expensive", "That's more than I can afford")

        # Get views for agents
        seller_messages = context.get_seller_view()  # Full compressed history
        buyer_history = context.get_buyer_view("lead_001")  # Only this lead's convo
    """

    # Priority levels for different event types
    PRIORITY_KEY_EVENT = 10  # Offers, decisions, call state changes
    PRIORITY_SELLER_MESSAGE = 5  # Seller's spoken messages
    PRIORITY_NORMAL = 0  # Regular tool outputs

    def __init__(
        self,
        seller_model_config: Optional[ModelConfig] = None,
        buyer_model_config: Optional[ModelConfig] = None,
        # Legacy parameters for backward compatibility
        max_tokens: int = 6000,
        compression_threshold: float = 0.8,
        buyer_max_tokens: int = 2000,
    ):
        """Initialize the episode context.

        Args:
            seller_model_config: ModelConfig for seller model (determines context limits).
            buyer_model_config: ModelConfig for buyer model (determines context limits).
            max_tokens: Legacy - Maximum tokens for seller context (used if no config).
            compression_threshold: Legacy - Trigger compression at this % of max_tokens.
            buyer_max_tokens: Legacy - Maximum tokens for each buyer's dialogue.
        """
        # Use provided model configs or create defaults
        if seller_model_config:
            self._seller_model_config = seller_model_config
        else:
            # Legacy fallback - create a config from the old parameters
            self._seller_model_config = ModelConfig(
                context_window=max_tokens + 4000,  # Add buffer for system tokens
                output_token_limit=2000,
                provider="unknown",
                compression_threshold=compression_threshold,
            )

        if buyer_model_config:
            self._buyer_model_config = buyer_model_config
        else:
            # Legacy fallback
            self._buyer_model_config = ModelConfig(
                context_window=buyer_max_tokens + 4000,
                output_token_limit=1000,
                provider="unknown",
                compression_threshold=compression_threshold,
            )

        self._lead_conversations: dict[str, LeadConversation] = {}

        # Create seller context manager
        self._seller_manager = SellerContextManager(self._seller_model_config)

        # Buyer managers created per-lead
        self._buyer_managers: dict[str, BuyerContextManager] = {}

        # Track current call for routing messages
        self._current_lead_id: Optional[str] = None
        self._message_counter = 0

        # Anchored state - critical info that survives compaction
        self._anchored_state = AnchoredState()

        # Legacy compatibility - expose the buffer through seller manager
        self._episode_buffer = self._seller_manager._buffer
        self._max_tokens = self._seller_model_config.available_context
        self._compression_threshold = self._seller_model_config.compression_threshold
        self._buyer_max_tokens = self._buyer_model_config.available_context

    def _get_lead_conversation(self, lead_id: str) -> LeadConversation:
        """Get or create conversation for a lead."""
        if lead_id not in self._lead_conversations:
            self._lead_conversations[lead_id] = LeadConversation(lead_id=lead_id)
        return self._lead_conversations[lead_id]

    # --- Recording events ---

    def record_seller_action(
        self,
        content: str,
        tool_calls: Optional[list] = None,
        is_spoken_message: bool = False,
    ) -> None:
        """Record seller's action (message and/or tool calls).

        Args:
            content: The seller's message or action description.
            tool_calls: Optional list of tool calls made.
            is_spoken_message: True if this is a message the buyer hears.
        """
        self._message_counter += 1

        # Determine priority
        priority = self.PRIORITY_SELLER_MESSAGE if is_spoken_message else self.PRIORITY_NORMAL

        # Build message content
        message_parts = []
        if content:
            message_parts.append(content)
        if tool_calls:
            tool_str = ", ".join(
                f"{tc.tool_name}({tc.arguments})" if hasattr(tc, "tool_name") else str(tc)
                for tc in tool_calls
            )
            message_parts.append(f"[Tools: {tool_str}]")

        full_content = "\n".join(message_parts) if message_parts else "[No content]"

        message = Message(
            role="assistant",
            content=full_content,
            priority=priority,
            timestamp=self._message_counter,
        )

        # Add to seller manager
        self._seller_manager.add(message)

        # If in a call and this is a spoken message, add to lead's conversation
        if self._current_lead_id and is_spoken_message and content:
            lead_convo = self._get_lead_conversation(self._current_lead_id)
            lead_convo.messages.append(message)
            # Add to dialogue-only list (just the spoken text, no tool info)
            lead_convo.dialogue_only.append(("seller", content))
            # Add to buyer manager for this lead
            buyer_manager = self._get_buyer_manager(self._current_lead_id)
            buyer_manager.add_dialogue("seller", content)

    def record_tool_result(
        self,
        tool_name: str,
        result: dict,
        lead_id: Optional[str] = None,
    ) -> None:
        """Record a tool execution result.

        Args:
            tool_name: Name of the tool executed.
            result: Result data from the tool.
            lead_id: Optional lead ID if tool is lead-specific.
        """
        self._message_counter += 1

        # ANCHOR search results - these survive compaction
        if tool_name == "crm.search_leads":
            # Record what was searched (even if no results)
            if filters := result.get("filters_applied"):
                self._anchored_state.record_search(filters)
            if "leads" in result:
                self._anchored_state.record_search_results(result["leads"], self._message_counter)

        # ANCHOR call starts
        if tool_name == "calling.start_call" and result.get("call_started"):
            if result_lead_id := result.get("lead_id"):
                lead_name = result.get("lead_name", "Unknown")
                self._anchored_state.record_call_started(result_lead_id, lead_name)

        # Determine priority based on tool type
        priority = self.PRIORITY_NORMAL

        # Format result for display
        result_str = self._format_tool_result(tool_name, result)

        message = Message(
            role="user",  # Tool results come as "observations"
            content=f"[{tool_name}]: {result_str}",
            priority=priority,
            timestamp=self._message_counter,
            metadata={"tool_name": tool_name, "lead_id": lead_id},
        )

        # Add to seller manager
        self._seller_manager.add(message)

    def record_offer(self, lead_id: str, offer: PlanOffer) -> None:
        """Record a plan offer presented to a lead.

        Args:
            lead_id: The lead who received the offer.
            offer: The plan offer presented.
        """
        self._message_counter += 1

        offer_dict = offer.to_dict() if hasattr(offer, "to_dict") else dict(offer)

        content = (
            f"[OFFER to {lead_id}] "
            f"Plan: {offer_dict.get('plan_id')}, "
            f"Premium: ${offer_dict.get('monthly_premium', 0):.2f}/mo, "
            f"Coverage: ${offer_dict.get('coverage_amount', 0):,.0f}"
        )

        # Key event - always preserved
        self._seller_manager.add_key_event(content, role="system")

        # Add to lead's conversation
        lead_convo = self._get_lead_conversation(lead_id)
        lead_convo.offers_presented.append(offer_dict)
        lead_convo.messages.append(
            Message(
                role="system",
                content=content,
                priority=self.PRIORITY_KEY_EVENT,
                timestamp=self._message_counter,
            )
        )

    def record_buyer_decision(
        self,
        lead_id: str,
        decision: str,
        dialogue: str,
        reason: str,
    ) -> None:
        """Record a buyer's decision.

        Args:
            lead_id: The lead who made the decision.
            decision: The decision (accept_plan, reject_plan, end_call).
            dialogue: What the buyer said.
            reason: The underlying reason for the decision.
        """
        self._message_counter += 1

        # ANCHOR the decision - survives compaction
        self._anchored_state.record_decision(lead_id, decision, self._message_counter)

        content = f'[{lead_id} {decision.upper()}]: "{dialogue}" (Reason: {reason})'

        # Key event - always preserved
        self._seller_manager.add_key_event(content, role="system")

        # Add to lead's conversation
        lead_convo = self._get_lead_conversation(lead_id)
        decision_record = {
            "decision": decision,
            "dialogue": dialogue,
            "reason": reason,
        }
        lead_convo.decisions.append(decision_record)
        lead_convo.messages.append(
            Message(
                role="user",  # Buyer's response
                content=f'Buyer: "{dialogue}"',
                priority=self.PRIORITY_KEY_EVENT,
                timestamp=self._message_counter,
            )
        )
        # Add to dialogue-only list (just what was spoken)
        if dialogue:
            lead_convo.dialogue_only.append(("buyer", dialogue))
            # Add to buyer manager for this lead
            buyer_manager = self._get_buyer_manager(lead_id)
            buyer_manager.add_dialogue("buyer", dialogue)

    def record_call_start(self, lead_id: str, lead_name: str = "Unknown") -> None:
        """Record the start of a call with a lead.

        Args:
            lead_id: The lead being called.
            lead_name: Name of the lead (for display in anchored state).
        """
        self._message_counter += 1
        self._current_lead_id = lead_id

        lead_convo = self._get_lead_conversation(lead_id)
        lead_convo.call_count += 1

        # Update anchored state with active call
        self._anchored_state.record_call_started(lead_id, lead_name)

        call_num = lead_convo.call_count
        content = f"[CALL START] Call #{call_num} with {lead_id}"

        # Key event - always preserved
        self._seller_manager.add_key_event(content, role="system")

        lead_convo.messages.append(
            Message(
                role="system",
                content=content,
                priority=self.PRIORITY_KEY_EVENT,
                timestamp=self._message_counter,
            )
        )

    def record_call_end(self, lead_id: str, reason: str) -> None:
        """Record the end of a call.

        Args:
            lead_id: The lead whose call ended.
            reason: Why the call ended.
        """
        self._message_counter += 1
        self._current_lead_id = None

        # Update anchored state - call ended
        self._anchored_state.record_call_ended(lead_id)

        content = f"[CALL END] Call with {lead_id} ended: {reason}"

        # Key event - always preserved
        self._seller_manager.add_key_event(content, role="system")

        lead_convo = self._get_lead_conversation(lead_id)
        lead_convo.messages.append(
            Message(
                role="system",
                content=content,
                priority=self.PRIORITY_KEY_EVENT,
                timestamp=self._message_counter,
            )
        )

    def _get_buyer_manager(self, lead_id: str) -> BuyerContextManager:
        """Get or create buyer manager for a lead."""
        if lead_id not in self._buyer_managers:
            self._buyer_managers[lead_id] = BuyerContextManager(self._buyer_model_config)
        return self._buyer_managers[lead_id]

    # --- Views ---

    def get_seller_view(self) -> list[dict]:
        """Get full episode history (compressed) for seller LLM.

        Injects anchored state at the start - always accurate, never lost to compaction.

        Returns:
            List of message dicts in API format.
        """
        # Prune old leads that were never called
        self._anchored_state.prune_old_leads(self._message_counter)

        messages = self._seller_manager.get_view()

        # INJECT anchored state - always accurate, never lost to compaction
        anchored_block = self._anchored_state.to_context_block()
        if anchored_block:
            messages.insert(
                0,
                {"role": "user", "content": f"[CURRENT STATE - Always Accurate]\n{anchored_block}"},
            )

        return messages

    def get_buyer_view(self, lead_id: str) -> str:
        """Get this lead's conversation history only (formatted as text).

        Uses BuyerContextManager for the lead if available, falls back to
        legacy _format_lead_history for existing conversations.

        Args:
            lead_id: The lead whose history to retrieve.

        Returns:
            Formatted string of this lead's conversation history.
        """
        # Use buyer manager if it exists for this lead
        if lead_id in self._buyer_managers:
            return self._buyer_managers[lead_id].get_view()

        # Legacy fallback for conversations started before managers were added
        if lead_id not in self._lead_conversations:
            return "This is the first interaction with this seller."

        convo = self._lead_conversations[lead_id]
        return self._format_lead_history(convo)

    def get_token_count(self) -> int:
        """Get current token count estimate for seller view."""
        return self._seller_manager.token_count()

    def get_seller_compression_trigger(self) -> int:
        """Get the seller's compression trigger threshold."""
        return self._seller_manager.compression_trigger

    @property
    def current_lead_id(self) -> Optional[str]:
        """Get the current lead being called, if any."""
        return self._current_lead_id

    # --- Compression ---

    def _maybe_compress(self) -> None:
        """Compress buffer when exceeding threshold.

        Delegates to seller manager which handles model-aware compression.
        """
        self._seller_manager.maybe_compress()

    def _summarize_messages(self, messages: list[Message]) -> str:
        """Summarize a list of messages for compression.

        Args:
            messages: Messages to summarize.

        Returns:
            Summary string.
        """
        # Group by tool types for summary
        tool_counts: dict[str, int] = {}
        lead_mentions: set[str] = set()

        for msg in messages:
            if msg.metadata.get("tool_name"):
                tool_name = msg.metadata["tool_name"]
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            if msg.metadata.get("lead_id"):
                lead_mentions.add(msg.metadata["lead_id"])

        parts = []
        if tool_counts:
            tool_str = ", ".join(f"{k}({v})" for k, v in sorted(tool_counts.items()))
            parts.append(f"Tool calls: {tool_str}")
        if lead_mentions:
            parts.append(f"Leads mentioned: {', '.join(sorted(lead_mentions))}")

        if parts:
            return f"[Summarized {len(messages)} earlier messages: {'; '.join(parts)}]"
        return f"[Summarized {len(messages)} earlier messages]"

    def _format_tool_result(self, tool_name: str, result: dict) -> str:
        """Format a tool result for display.

        Args:
            tool_name: Name of the tool.
            result: Result data.

        Returns:
            Formatted string.
        """
        # Truncate long results
        import json

        try:
            result_str = json.dumps(result, default=str)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            return result_str
        except Exception:
            return str(result)[:500]

    def _estimate_dialogue_tokens(self, dialogue: list[tuple[str, str]]) -> int:
        """Estimate token count for dialogue list.

        Args:
            dialogue: List of (role, text) tuples.

        Returns:
            Estimated token count.
        """
        total_chars = sum(len(text) + 15 for _, text in dialogue)  # +15 for role prefix overhead
        return total_chars // 4  # ~4 chars per token

    def _format_lead_history(self, convo: LeadConversation) -> str:
        """Format a lead's conversation history for the buyer.

        Only shows actual dialogue text - what the seller said and what the buyer said.
        No system messages, CRM searches, or internal tool info.
        Compresses when token limit is approached (consistent with seller compression).

        Args:
            convo: The lead's conversation record.

        Returns:
            Formatted history string with only dialogue.
        """
        if not convo.dialogue_only and not convo.offers_presented:
            return "This is the first interaction with this seller."

        lines = []

        # Note if this is a repeat call
        if convo.call_count > 1:
            lines.append(f"This is call #{convo.call_count} with this seller.")
            lines.append("")

        # Get dialogue - compress if exceeding token threshold
        dialogue = list(convo.dialogue_only)
        threshold = self._buyer_max_tokens * self._compression_threshold

        if self._estimate_dialogue_tokens(dialogue) > threshold:
            # Remove older turns until under threshold, keeping at least recent messages
            older_turns = []
            original_tokens = self._estimate_dialogue_tokens(dialogue)
            while (
                len(dialogue) > 4  # Keep at least 4 recent turns
                and self._estimate_dialogue_tokens(dialogue) > threshold
            ):
                older_turns.append(dialogue.pop(0))

            if older_turns:
                # Summarize what was removed
                seller_count = sum(1 for role, _ in older_turns if role == "seller")
                buyer_count = sum(1 for role, _ in older_turns if role == "buyer")
                print(
                    f"  [Context] Summarizing buyer context ({convo.lead_id}): {original_tokens} tokens > {threshold:.0f} threshold, "
                    f"compressed {len(older_turns)} turns"
                )
                lines.append(
                    f"[Earlier in conversation: {seller_count} seller messages, {buyer_count} of your responses - summarized]"
                )
                lines.append("")

        # Format the dialogue
        if dialogue:
            lines.append("Conversation so far:")
            for role, text in dialogue:
                if role == "seller":
                    lines.append(f'  Seller: "{text}"')
                else:
                    lines.append(f'  You: "{text}"')

        # Summarize decisions made (without internal details)
        if convo.decisions:
            lines.append("")
            rejected = sum(1 for d in convo.decisions if d.get("decision") == "reject_plan")
            if rejected > 0:
                lines.append(f"You have rejected {rejected} offer(s) from this seller.")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all context for a new episode."""
        self._lead_conversations.clear()
        self._buyer_managers.clear()
        # Create fresh seller manager
        self._seller_manager = SellerContextManager(self._seller_model_config)
        self._episode_buffer = self._seller_manager._buffer
        self._current_lead_id = None
        self._message_counter = 0
        # Reset anchored state
        self._anchored_state = AnchoredState()
