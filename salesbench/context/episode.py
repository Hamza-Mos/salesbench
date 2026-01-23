"""Episode-wide context management for conversation history.

Manages conversation context for both seller (episode-wide) and buyer (per-lead) agents
with model-aware compression based on context window sizes.

Architecture:
- Seller: Episode-wide context - sees ALL conversations with ALL leads
- Buyer: Per-lead context - sees ONLY their own conversation with the seller

Context managers use ModelConfig to determine compression thresholds dynamically
based on each model's actual context window size.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

from salesbench.context.anchored_state import (
    AnchoredState,
    DecisionRecord,
    LeadSummary,
)
from salesbench.context.buffers import LLMCompactBuffer, Message
from salesbench.core.types import PlanOffer
from salesbench.models import ModelConfig

# Re-export for backwards compatibility
__all__ = [
    "AnchoredState",
    "LeadSummary",
    "DecisionRecord",
    "LeadConversation",
    "BaseContextManager",
    "SellerContextManager",
    "BuyerContextManager",
    "EpisodeContext",
]


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


class BaseContextManager(ABC):
    """Base class for model-aware context management."""

    def __init__(self, model_config: ModelConfig):
        """Initialize with model configuration.

        Args:
            model_config: Configuration for the model (context window, etc.)
        """
        self.model_config = model_config

    @property
    def compression_trigger(self) -> int:
        """Token count at which to trigger compression."""
        return self.model_config.compression_trigger

    @abstractmethod
    async def maybe_compress(self) -> bool:
        """Compress if threshold exceeded.

        Returns:
            True if compaction was performed, False otherwise.
        """
        pass

    @abstractmethod
    def token_count(self) -> int:
        """Get current token count estimate."""
        pass


class SellerContextManager(BaseContextManager):
    """Manages seller's episode-wide context with LLM compaction.

    Seller sees ALL conversations with ALL leads.
    Uses LLMCompactBuffer for LLM-based summarization when context exceeds threshold.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        compaction_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        self.model_config = model_config
        # Use LLMCompactBuffer with LLM-based compaction
        self._buffer = LLMCompactBuffer(
            max_tokens=model_config.available_context,
            keep_recent=model_config.compaction_keep_recent,
            compaction_fn=compaction_fn,
        )
        self._lead_conversations: dict[str, LeadConversation] = {}

    @property
    def compression_trigger(self) -> int:
        """Token count at which to trigger compression."""
        return self.model_config.compression_trigger

    async def maybe_compress(self) -> bool:
        """Compress when exceeding model's context threshold using LLM compaction.

        Returns:
            True if compaction was performed, False otherwise.
        """
        return await self._buffer.compact_if_needed(self.compression_trigger)

    def get_view(self) -> list[dict]:
        """Get seller's full episode history.

        Note: Call maybe_compress() before this if async compaction is needed.

        Returns:
            List of message dicts in API format, with optional memory prefix.
        """
        messages = self._buffer.to_api_messages()

        # Inject compacted memory if exists
        memory = self._buffer.get_memory_prefix()
        if memory:
            messages.insert(
                0,
                {"role": "user", "content": f"[MEMORY - What You Remember]\n{memory}"},
            )

        return messages

    def add(self, message: Message) -> None:
        """Add a message to the buffer."""
        self._buffer.add(message)

    def add_key_event(self, content: str, role: str = "system") -> None:
        """Add a key event message."""
        self._buffer.add_key_event(content, role=role)

    def token_count(self) -> int:
        """Get current token count estimate."""
        return self._buffer.token_count()


class BuyerContextManager(BaseContextManager):
    """Manages buyer's per-lead context with LLM compaction.

    Buyer sees ONLY their own conversation with seller.
    Uses LLM-based compaction when context exceeds threshold.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        compaction_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        super().__init__(model_config)
        self._dialogue: list[tuple[str, str]] = []  # (role, text)
        self._compacted_memory: Optional[str] = None
        self._compaction_fn = compaction_fn
        self.keep_recent = model_config.compaction_keep_recent

    async def maybe_compress(self) -> bool:
        """Compress dialogue when exceeding threshold using LLM compaction.

        Returns:
            True if compaction was performed, False otherwise.
        """
        current_tokens = self._estimate_tokens()
        if current_tokens <= self.compression_trigger:
            return False

        if len(self._dialogue) <= self.keep_recent:
            return False

        before_turns = len(self._dialogue)
        older = self._dialogue[: -self.keep_recent]
        recent = self._dialogue[-self.keep_recent :]

        logger.info(
            f"[Buyer Context] COMPACTING: {current_tokens:,} > {self.compression_trigger:,} tokens "
            f"({before_turns} turns)"
        )

        if self._compaction_fn:
            older_text = self._format_dialogue(older)
            self._compacted_memory = await self._compaction_fn(older_text)

        self._dialogue = recent
        after_tokens = self._estimate_tokens()

        logger.info(
            f"[Buyer Context] COMPACTED via LLM: {after_tokens:,} tokens "
            f"({len(recent)} turns kept, {len(older)} turns summarized)"
        )
        return True

    def _format_dialogue(self, dialogue: list[tuple[str, str]]) -> str:
        """Format dialogue for compaction prompt."""
        lines = []
        for role, text in dialogue:
            speaker = "Seller" if role == "seller" else "You"
            lines.append(f'{speaker}: "{text}"')
        return "\n".join(lines)

    def _estimate_tokens(self) -> int:
        """Estimate token count for current dialogue.

        Uses consistent formula: len(content) // 4 + 10 (same as Message.token_estimate).
        """
        base = sum(len(text) // 4 + 10 for _, text in self._dialogue)
        if self._compacted_memory:
            base += len(self._compacted_memory) // 4 + 10
        return base

    def add_dialogue(self, role: str, text: str) -> None:
        """Add a dialogue turn."""
        self._dialogue.append((role, text))

    def get_view(self) -> str:
        """Get buyer's conversation history as formatted text.

        Note: Call maybe_compress() before this if async compaction is needed.

        Returns:
            Formatted string of conversation history with optional memory prefix.
        """
        lines = []
        if self._compacted_memory:
            lines.append("## What You Remember From Earlier")
            lines.append(self._compacted_memory)
            lines.append("")

        if self._dialogue:
            lines.append("## Recent Conversation")
            for role, text in self._dialogue:
                speaker = "Seller" if role == "seller" else "You"
                lines.append(f'  {speaker}: "{text}"')
        elif not self._compacted_memory:
            lines.append("This is the first interaction with this seller.")

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
        seller_model_config: ModelConfig,
        buyer_model_config: ModelConfig,
        seller_compaction_fn: Optional[Callable[[str], Awaitable[str]]] = None,
        buyer_compaction_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    ):
        """Initialize the episode context.

        Args:
            seller_model_config: ModelConfig for seller model (determines context limits).
            buyer_model_config: ModelConfig for buyer model (determines context limits).
            seller_compaction_fn: Optional async function for seller LLM compaction.
            buyer_compaction_fn: Optional async function for buyer LLM compaction.
        """
        self._seller_model_config = seller_model_config
        self._buyer_model_config = buyer_model_config
        self._seller_compaction_fn = seller_compaction_fn
        self._buyer_compaction_fn = buyer_compaction_fn

        self._lead_conversations: dict[str, LeadConversation] = {}

        # Create seller context manager with compaction function
        self._seller_manager = SellerContextManager(
            self._seller_model_config,
            compaction_fn=seller_compaction_fn,
        )

        # Buyer managers created per-lead with buyer compaction function
        self._buyer_managers: dict[str, BuyerContextManager] = {}

        # Track current call for routing messages
        self._current_lead_id: Optional[str] = None
        self._message_counter = 0

        # Anchored state - critical info that survives compaction
        self._anchored_state = AnchoredState()

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
        gemini_content: Optional[Any] = None,
    ) -> None:
        """Record seller's action (message and/or tool calls).

        Args:
            content: The seller's message or action description.
            tool_calls: Optional list of tool calls made.
            is_spoken_message: True if this is a message the buyer hears.
            gemini_content: Raw Gemini Content object for thought_signature preservation.
                           Required for Gemini 3 multi-turn function calling.
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

        # Build metadata - include gemini_content if provided
        metadata = {}
        if gemini_content is not None:
            metadata["gemini_content"] = gemini_content

        message = Message(
            role="assistant",
            content=full_content,
            priority=priority,
            timestamp=self._message_counter,
            metadata=metadata,
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

    def get_last_seller_utterance(self, lead_id: str) -> Optional[str]:
        """Return the most recent seller-spoken utterance for a lead, if any."""
        convo = self._lead_conversations.get(lead_id)
        if not convo or not convo.dialogue_only:
            return None
        for role, text in reversed(convo.dialogue_only):
            if role == "seller" and text:
                return text
        return None

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

    def record_buyer_message(self, lead_id: str, dialogue: str) -> None:
        """Record a buyer's conversational message (NOT a decision).

        This should be used for normal back-and-forth dialogue during a call,
        when the buyer is responding naturally but has not accepted/rejected/ended.

        IMPORTANT: This does NOT update AnchoredState decisions and must NOT
        end calls or change lead availability.
        """
        if not dialogue:
            return

        self._message_counter += 1

        lead_convo = self._get_lead_conversation(lead_id)

        # Store as a normal user message in the per-lead conversation
        lead_convo.messages.append(
            Message(
                role="user",
                content=f'Buyer: "{dialogue}"',
                priority=self.PRIORITY_SELLER_MESSAGE,
                timestamp=self._message_counter,
            )
        )

        # Dialogue-only + buyer manager (for per-lead history given back to buyer simulator)
        lead_convo.dialogue_only.append(("buyer", dialogue))
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
            self._buyer_managers[lead_id] = BuyerContextManager(
                self._buyer_model_config,
                compaction_fn=self._buyer_compaction_fn,
            )
        return self._buyer_managers[lead_id]

    # --- Views ---

    def get_seller_view(self) -> list[dict]:
        """Get full episode history for seller LLM.

        Note: Call trigger_seller_compaction() before this for async compaction.

        Injects anchored state at the start - always accurate, never lost to compaction.

        Returns:
            List of message dicts in API format.
        """
        # Prune old leads that were never called
        self._anchored_state.prune_old_leads(self._message_counter)

        messages = self._seller_manager.get_view()
        token_count = self._seller_manager.token_count()

        logger.debug(f"[Seller View] Retrieved {len(messages)} messages, ~{token_count:,} tokens")

        # INJECT anchored state - always accurate, never lost to compaction
        anchored_block = self._anchored_state.to_context_block()
        if anchored_block:
            messages.insert(
                0,
                {"role": "user", "content": f"[CURRENT STATE - Always Accurate]\n{anchored_block}"},
            )
            logger.debug(
                f"[Seller View] Injected AnchoredState: "
                f"{len(self._anchored_state.found_leads)} leads, "
                f"{len(self._anchored_state.called_lead_ids)} called, "
                f"{len(self._anchored_state.accepted_lead_ids)} accepted, "
                f"active_call={self._anchored_state.active_call_lead_id}"
            )

        return messages

    async def trigger_seller_compaction(self) -> bool:
        """Trigger async compaction for seller's context if needed.

        Call this before get_seller_view() when async compaction is required.

        Returns:
            True if compaction was performed, False otherwise.
        """
        return await self._seller_manager.maybe_compress()

    def trigger_seller_compaction_sync(self) -> None:
        """Trigger seller compaction from synchronous code.

        Use this when calling from sync context (e.g., seller agent).
        """
        try:
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                asyncio.create_task(self.trigger_seller_compaction())
        except RuntimeError:
            # Not in an event loop - skip compaction (it's optional optimization)
            pass

    def get_buyer_structured_context(self, lead_id: str) -> str:
        """Extract structured decision history for a lead.

        This provides a summary of rejections that survives context compression,
        ensuring the buyer remembers their previous objections.

        Args:
            lead_id: The lead whose decision history to retrieve.

        Returns:
            Formatted string summarizing past decisions, or empty string if none.
        """
        lead_convo = self._lead_conversations.get(lead_id)
        if not lead_convo:
            return ""

        # Count rejections and collect rejection details
        rejections = [d for d in lead_convo.decisions if d.get("decision") == "reject_plan"]
        if not rejections:
            return ""

        lines = ["## Your Decision History This Call"]
        lines.append(f"- You have rejected {len(rejections)} offer(s) so far")

        # Include rejected offer details if we have them
        if lead_convo.offers_presented and rejections:
            lines.append("- Rejected offers:")
            # Match rejections with offers (they should align)
            for i, rejection in enumerate(rejections):
                reason = rejection.get("reason", "No specific reason")
                dialogue = rejection.get("dialogue", "")
                # Get corresponding offer if available
                if i < len(lead_convo.offers_presented):
                    offer = lead_convo.offers_presented[i]
                    plan_type = offer.get("plan_id", "Unknown")
                    premium = offer.get("monthly_premium", 0)
                    lines.append(f'  * {plan_type} ${premium:.0f}/mo - "{dialogue or reason}"')
                else:
                    lines.append(f'  * Offer #{i+1} - "{dialogue or reason}"')

        return "\n".join(lines)

    def get_buyer_view(self, lead_id: str) -> str:
        """Get this lead's conversation history only (formatted as text).

        Note: Call trigger_buyer_compaction() before this for async compaction.

        Uses BuyerContextManager for the lead. Prepends structured decision history
        to ensure the buyer remembers their objections even after context compression.

        Args:
            lead_id: The lead whose history to retrieve.

        Returns:
            Formatted string of this lead's conversation history.
        """
        # Get structured decision context (survives compression)
        structured_context = self.get_buyer_structured_context(lead_id)

        # Get or create buyer manager for this lead
        buyer_manager = self._get_buyer_manager(lead_id)
        dialogue_view = buyer_manager.get_view()

        # Combine: structured decisions first, then dialogue
        if structured_context:
            return f"{structured_context}\n\n{dialogue_view}"
        return dialogue_view

    async def trigger_buyer_compaction(self, lead_id: str) -> bool:
        """Trigger async compaction for a buyer's context if needed.

        Call this before get_buyer_view() when async compaction is required.

        Args:
            lead_id: The lead whose context to potentially compact.

        Returns:
            True if compaction was performed, False otherwise.
        """
        buyer_manager = self._get_buyer_manager(lead_id)
        return await buyer_manager.maybe_compress()

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

    def _format_tool_result(self, tool_name: str, result: dict) -> str:
        """Format a tool result for display.

        Args:
            tool_name: Name of the tool.
            result: Result data.

        Returns:
            Formatted string.
        """
        # Truncate long results
        try:
            result_str = json.dumps(result, default=str)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            return result_str
        except Exception:
            return str(result)[:500]

    def reset(self) -> None:
        """Reset all context for a new episode."""
        logger.info("[EpisodeContext] Resetting for new episode")
        self._lead_conversations.clear()
        self._buyer_managers.clear()
        # Create fresh seller manager with compaction function
        self._seller_manager = SellerContextManager(
            self._seller_model_config,
            compaction_fn=self._seller_compaction_fn,
        )
        self._current_lead_id = None
        self._message_counter = 0
        # Reset anchored state
        self._anchored_state = AnchoredState()
        logger.debug(
            f"[EpisodeContext] Reset complete - "
            f"Seller trigger: {self._seller_model_config.compression_trigger:,} tokens"
        )
