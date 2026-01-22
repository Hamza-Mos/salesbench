"""Calling tools for the sales environment.

Tools:
- calling.start_call: Start a call with a lead
- calling.propose_plan: Propose an insurance plan (invokes buyer)
- calling.end_call: End the current call

This is where buyer simulation is invoked.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

from salesbench.core.config import BudgetConfig

logger = logging.getLogger(__name__)
from salesbench.agents.buyer_llm import BuyerSimulatorFn
from salesbench.core.types import (
    BuyerDecision,
    BuyerResponseData,
    CallSession,
    LeadID,
    LeadStatus,
    NextStep,
    PlanOffer,
    PlanType,
    ToolResult,
    generate_call_id,
)

if TYPE_CHECKING:
    from salesbench.context.episode import EpisodeContext
    from salesbench.envs.sales_mvp.state import EnvironmentState


class CallingTools:
    """Calling tool implementations."""

    def __init__(
        self,
        state: "EnvironmentState",
        budget: BudgetConfig,
        buyer_simulator: Optional[BuyerSimulatorFn] = None,
    ):
        self.state = state
        self.budget = budget
        self._buyer_simulator = buyer_simulator
        self._episode_context: Optional["EpisodeContext"] = None

    def set_buyer_simulator(self, simulator: BuyerSimulatorFn) -> None:
        """Set the buyer simulator callback."""
        self._buyer_simulator = simulator

    def set_episode_context(self, context: "EpisodeContext") -> None:
        """Set the episode context for conversation history.

        Args:
            context: The episode context to use for buyer history.
        """
        self._episode_context = context

    def _trigger_buyer_compaction(self, lead_id: str) -> None:
        """Trigger async buyer compaction from sync context.

        Runs the async compaction method using asyncio.
        """
        if not self._episode_context:
            return

        try:
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                asyncio.create_task(self._episode_context.trigger_buyer_compaction(lead_id))
        except RuntimeError:
            # Not in an event loop - skip compaction (it's optional optimization)
            pass

    def start_call(self, lead_id: str) -> ToolResult:
        """Start a call with a lead.

        Args:
            lead_id: The lead to call.

        Returns:
            ToolResult with call session info.
        """
        # Check for active call
        if self.state.active_call is not None:
            logger.warning(f"[SELLER:calling.start_call] Failed - call already active: {self.state.active_call.call_id}")
            return ToolResult(
                call_id="error",
                success=False,
                error=f"Call already active: {self.state.active_call.call_id}. End it first.",
            )

        # No artificial daily call limit - time naturally constrains throughput
        # (Each call costs time, so you can only make ~80-100 calls in 8 hours)

        # Get lead
        lead = self.state.get_lead(LeadID(lead_id))
        if not lead:
            logger.warning(f"[SELLER:calling.start_call] Failed - lead not found: {lead_id}")
            return ToolResult(
                call_id="error",
                success=False,
                error=f"Lead not found: {lead_id}",
            )

        # Check lead status - only ACTIVE leads can be called
        if lead.status == LeadStatus.DNC or lead.on_dnc_list:
            self.state.stats.dnc_violations += 1
            logger.error(f"[SELLER:calling.start_call] DNC VIOLATION - attempted to call {lead.name} (ID: {lead_id})")
            return ToolResult(
                call_id="error",
                success=False,
                error=f"DNC VIOLATION: Lead {lead_id} is on Do Not Call list",
                data={"dnc_violation": True, "lead_status": lead.status.value},
            )

        if lead.status == LeadStatus.CONVERTED or lead.converted:
            logger.info(f"[SELLER:calling.start_call] Failed - lead {lead.name} already converted")
            return ToolResult(
                call_id="error",
                success=False,
                error=f"Lead {lead_id} already converted (accepted previously).",
                data={"converted": True, "lead_status": lead.status.value},
            )

        # Create call session
        call_id = generate_call_id()
        session = CallSession(
            call_id=call_id,
            lead_id=LeadID(lead_id),
            started_at=self.state.time.total_minutes(),
        )
        self.state.active_call = session
        self.state.stats.total_calls += 1

        # Advance time (configurable cost to connect)
        self.state.time.advance_minutes(self.budget.start_call_cost, self.budget)

        logger.info(f"[SELLER:calling.start_call] Started call with {lead.name} (ID: {lead_id}, temp: {lead.temperature.value})")

        return ToolResult(
            call_id="",
            success=True,
            data={
                "call_id": call_id,
                "lead_id": lead_id,
                "lead_name": lead.name,
                "lead_info": {
                    "age": lead.age,
                    "job": lead.job,
                    "income": lead.annual_income,
                    "household_size": lead.household_size,
                    "trigger": lead.trigger,
                    "temperature": lead.temperature.value,
                },
                "call_started": True,
                "status": "connected",
                "message": f"Call started with {lead.name}. You may now present plans.",
            },
        )

    def propose_plan(
        self,
        plan_id: str,
        monthly_premium: float,
        coverage_amount: float,
        next_step: str,
        term_years: Optional[int] = None,
    ) -> ToolResult:
        """Record a plan offer for analytics.

        This is purely analytical - it records the structured offer data
        for tracking purposes. The actual conversation with the buyer
        happens through the seller's message text.

        Args:
            plan_id: The plan type (TERM, WHOLE, UL, VUL, LTC, DI).
            monthly_premium: Monthly premium amount.
            coverage_amount: Coverage amount.
            next_step: Proposed next step (schedule_followup, request_info, close_now).
            term_years: Term length for TERM plans.

        Returns:
            ToolResult confirming the offer was recorded.
        """
        # Check for active call
        if self.state.active_call is None:
            return ToolResult(
                call_id="error",
                success=False,
                error="No active call. Start a call first.",
            )

        session = self.state.active_call

        # Validate plan_id
        try:
            plan_type = PlanType(plan_id)
        except ValueError:
            return ToolResult(
                call_id="error",
                success=False,
                error=f"Invalid plan_id: {plan_id}. Valid: {[p.value for p in PlanType]}",
            )

        # Validate next_step
        try:
            next_step_enum = NextStep(next_step)
        except ValueError:
            return ToolResult(
                call_id="error",
                success=False,
                error=f"Invalid next_step: {next_step}. Valid: {[n.value for n in NextStep]}",
            )

        # Basic bounds checking
        if monthly_premium <= 0 or monthly_premium > 10000:
            return ToolResult(
                call_id="error",
                success=False,
                error=f"Invalid monthly_premium: {monthly_premium}. Must be between $1 and $10,000.",
            )
        if coverage_amount <= 0 or coverage_amount > 10_000_000:
            return ToolResult(
                call_id="error",
                success=False,
                error=f"Invalid coverage_amount: {coverage_amount}. Must be between $1 and $10M.",
            )

        # Create offer
        offer = PlanOffer(
            plan_id=plan_type,
            monthly_premium=monthly_premium,
            next_step=next_step_enum,
            coverage_amount=coverage_amount,
            term_years=term_years,
        )

        # Get lead
        lead = self.state.get_lead(session.lead_id)
        if not lead:
            return ToolResult(
                call_id="error",
                success=False,
                error=f"Lead not found: {session.lead_id}",
            )

        # Invoke buyer simulator (LLM required)
        if not self._buyer_simulator:
            return ToolResult(
                call_id="error",
                success=False,
                error="Buyer simulator not configured. LLM buyer is required.",
            )

        # Hard stop: If patience is exhausted (<=5%), buyer hangs up automatically
        if lead.hidden.patience <= 0.05:
            logger.info(
                f"[Patience Exhausted] {lead.name} patience at {lead.hidden.patience:.0%} - automatic hang up"
            )
            buyer_response = BuyerResponseData(
                decision=BuyerDecision.END_CALL,
                reason="Patience exhausted",
                dialogue="I've had enough. I really need to go now.",
                request_dnc=lead.hidden.patience <= 0.0,  # Request DNC if patience hit 0
            )
            # Skip LLM call - go directly to handling the response
            session.offers_presented.append(offer)
            session.buyer_responses.append(buyer_response)
            self.state.time.advance_minutes(1, self.budget)  # Quick hang up
            session.duration_minutes += 1
            session.outcome = BuyerDecision.END_CALL
            self.state.record_call_outcome(BuyerDecision.END_CALL)
            self._finalize_call(session)

            result_data = {
                "decision": buyer_response.decision.value,
                "reason": buyer_response.reason,
                "dialogue": buyer_response.dialogue,
                "offer_presented": offer.to_dict(),
                "offers_so_far": len(session.offers_presented),
                "message": "Buyer ENDED the call (patience exhausted).",
                "call_ended": True,
            }
            if buyer_response.request_dnc:
                lead.status = LeadStatus.DNC
                lead.on_dnc_list = True
                result_data["dnc_requested"] = True
                result_data["message"] += " Lead requested Do Not Call."
                logger.info(f"[Patience Exhausted] {lead.name} added to DNC (patience at 0%)")

            return ToolResult(call_id="", success=True, data=result_data)

        # Get negotiation history for this lead from episode context
        negotiation_history = None
        seller_pitch = None
        if self._episode_context:
            # Trigger async compaction before getting buyer view
            self._trigger_buyer_compaction(str(session.lead_id))
            negotiation_history = self._episode_context.get_buyer_view(str(session.lead_id))
            seller_pitch = self._episode_context.get_last_seller_utterance(str(session.lead_id))

        # Condition the buyer's decision on what the seller actually said most recently.
        buyer_response = self._buyer_simulator(
            lead, offer, session, seller_pitch, negotiation_history
        )

        # Record offer and response
        session.offers_presented.append(offer)
        session.buyer_responses.append(buyer_response)

        # Advance time (configurable cost per offer presentation)
        cost = self.budget.propose_plan_cost
        self.state.time.advance_minutes(cost, self.budget)
        session.duration_minutes += int(cost)

        # Handle decision consequences
        result_data = {
            "decision": buyer_response.decision.value,
            "reason": buyer_response.reason,
            "dialogue": buyer_response.dialogue,
            "offer_presented": offer.to_dict(),
            "offers_so_far": len(session.offers_presented),
        }

        if buyer_response.decision == BuyerDecision.ACCEPT_PLAN:
            session.outcome = BuyerDecision.ACCEPT_PLAN
            self.state.record_call_outcome(BuyerDecision.ACCEPT_PLAN)
            result_data["message"] = "Buyer ACCEPTED the plan! End the call with calling.end_call to finalize."
            # Mark lead converted (updates both status and legacy flag)
            lead.status = LeadStatus.CONVERTED
            lead.converted = True
            # NOTE: Call NOT auto-ended - seller must call end_call themselves
            logger.info(f"[BUYER:ACCEPT_PLAN] {lead.name} accepted offer: ${offer.monthly_premium}/mo for ${offer.coverage_amount} coverage")

        elif buyer_response.decision == BuyerDecision.REJECT_PLAN:
            self.state.record_call_outcome(BuyerDecision.REJECT_PLAN)
            # Track rejection count for analytics
            lead.rejection_count += 1

            # Decay patience on rejection (models real human frustration)
            patience_before = lead.hidden.patience
            base_decay = 0.12  # ~12% per rejection
            if lead.rejection_count >= 3:
                base_decay = 0.18  # More decay after multiple rejections
            lead.hidden.patience = max(0.0, lead.hidden.patience - base_decay)
            logger.info(
                f"[Patience Decay] {lead.name} patience {patience_before:.0%} -> {lead.hidden.patience:.0%} "
                f"after rejection #{lead.rejection_count}"
            )

            result_data["message"] = (
                f"Buyer REJECTED this offer ({lead.rejection_count} rejections). "
                "You may present another plan or try a different approach."
            )
            result_data["rejection_count"] = lead.rejection_count
            result_data["patience_remaining"] = f"{lead.hidden.patience:.0%}"
            logger.info(f"[BUYER:REJECT_PLAN] {lead.name} rejected offer ({lead.rejection_count} rejections)")

        elif buyer_response.decision == BuyerDecision.END_CALL:
            session.outcome = BuyerDecision.END_CALL
            self.state.record_call_outcome(BuyerDecision.END_CALL)
            result_data["message"] = "Buyer ENDED the call (hung up)."
            result_data["call_ended"] = True
            logger.info(f"[BUYER:END_CALL] {lead.name} ended the call (hung up)")

            # Buyer hung up - call is over (this is realistic, not handholding)
            self._finalize_call(session)

        # Check if buyer explicitly requested DNC (buyer's own decision, no triggers)
        if buyer_response.request_dnc:
            lead.status = LeadStatus.DNC
            lead.on_dnc_list = True
            result_data["dnc_requested"] = True
            result_data["message"] += " Lead requested Do Not Call."
            logger.info(f"[BUYER:REQUEST_DNC] {lead.name} requested Do Not Call - added to DNC list")

        # Add patience warning when patience is running low
        if lead.hidden.patience <= 0.20:
            result_data["patience_warning"] = "Buyer patience is running low"

        return ToolResult(
            call_id="",
            success=True,
            data=result_data,
        )

    def end_call(self, reason: Optional[str] = None) -> ToolResult:
        """End the current call.

        Args:
            reason: Optional reason for ending.

        Returns:
            ToolResult with call summary.
        """
        if self.state.active_call is None:
            logger.warning("[SELLER:calling.end_call] Failed - no active call to end")
            return ToolResult(
                call_id="error",
                success=False,
                error="No active call to end.",
            )

        session = self.state.active_call
        lead = self.state.get_lead(session.lead_id)
        lead_name = lead.name if lead else "Unknown"

        self._finalize_call(session)

        logger.info(f"[SELLER:calling.end_call] Ended call with {lead_name} (duration: {session.duration_minutes}min, offers: {len(session.offers_presented)})")

        return ToolResult(
            call_id="",
            success=True,
            data={
                "call_id": session.call_id,
                "lead_id": session.lead_id,
                "lead_name": lead.name if lead else "Unknown",
                "duration_minutes": session.duration_minutes,
                "offers_presented": len(session.offers_presented),
                "outcome": session.outcome.value if session.outcome else "no_outcome",
                "reason": reason,
            },
        )

    def _finalize_call(self, session: CallSession) -> None:
        """Finalize a call session."""
        session.ended_at = self.state.time.total_minutes()
        session.duration_minutes = max(1, session.ended_at - session.started_at)
        self.state.stats.total_call_minutes += session.duration_minutes
        self.state.call_history.append(session)
        self.state.active_call = None


    def get_buyer_conversational_response(self, seller_message: str) -> Optional[str]:
        """Get a conversational response from the buyer (not a decision).

        This is called when the seller speaks while in a call.
        The buyer responds naturally based on their persona.

        Args:
            seller_message: What the salesperson just said.

        Returns:
            The buyer's dialogue response, or None if not in a call or no simulator.
        """
        # Must be in a call
        if self.state.active_call is None:
            return None

        # Need a buyer simulator
        if not self._buyer_simulator:
            return None

        session = self.state.active_call
        lead = self.state.get_lead(session.lead_id)
        if not lead:
            return None

        # Get conversation history (with compaction)
        negotiation_history = None
        if self._episode_context:
            self._trigger_buyer_compaction(str(session.lead_id))
            negotiation_history = self._episode_context.get_buyer_view(str(session.lead_id))

        # Check if buyer simulator has the conversational response method
        if hasattr(self._buyer_simulator, "get_conversational_response"):
            return self._buyer_simulator.get_conversational_response(
                lead, seller_message, session, negotiation_history
            )

        # Fallback: no conversational support
        return None

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a calling tool.

        Args:
            tool_name: Full tool name (e.g., "calling.start_call").
            arguments: Tool arguments.

        Returns:
            ToolResult from the tool execution.
        """
        method_name = tool_name.replace("calling.", "")

        if method_name == "start_call":
            if "lead_id" not in arguments:
                return ToolResult(
                    call_id="error",
                    success=False,
                    error="Missing required argument: lead_id",
                )
            return self.start_call(arguments["lead_id"])

        elif method_name == "propose_plan":
            required = ["plan_id", "monthly_premium", "coverage_amount", "next_step"]
            missing = [r for r in required if r not in arguments]
            if missing:
                return ToolResult(
                    call_id="error",
                    success=False,
                    error=f"Missing required arguments: {missing}",
                )
            return self.propose_plan(
                plan_id=arguments["plan_id"],
                monthly_premium=arguments["monthly_premium"],
                coverage_amount=arguments["coverage_amount"],
                next_step=arguments["next_step"],
                term_years=arguments.get("term_years"),
            )

        elif method_name == "end_call":
            return self.end_call(reason=arguments.get("reason"))

        else:
            return ToolResult(
                call_id="error",
                success=False,
                error=f"Unknown calling tool: {tool_name}",
            )
