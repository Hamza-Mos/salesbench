"""Calling tools for the sales environment.

Tools:
- calling.start_call: Start a call with a lead
- calling.propose_plan: Propose an insurance plan (invokes buyer)
- calling.end_call: End the current call

This is where buyer simulation is invoked.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

from salesbench.core.config import BudgetConfig
from salesbench.core.types import (
    BuyerDecision,
    BuyerResponseData,
    CallSession,
    LeadID,
    NextStep,
    PlanOffer,
    PlanType,
    ToolResult,
    generate_call_id,
)

if TYPE_CHECKING:
    from salesbench.context.episode import EpisodeContext
    from salesbench.envs.sales_mvp.personas import Persona
    from salesbench.envs.sales_mvp.state import EnvironmentState


# Type for buyer simulator callback (persona, offer, session, seller_pitch, negotiation_history)
BuyerSimulatorFn = Callable[
    ["Persona", PlanOffer, "CallSession", Optional[str], Optional[str]],
    BuyerResponseData,
]


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

    def start_call(self, lead_id: str) -> ToolResult:
        """Start a call with a lead.

        Args:
            lead_id: The lead to call.

        Returns:
            ToolResult with call session info.
        """
        # Check for active call
        if self.state.active_call is not None:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Call already active: {self.state.active_call.call_id}. End it first.",
            )

        # Check daily call limit
        if self.state.stats.calls_today >= self.budget.max_calls_per_day:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Daily call limit reached: {self.budget.max_calls_per_day}",
            )

        # Get lead
        lead = self.state.get_lead(LeadID(lead_id))
        if not lead:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Lead not found: {lead_id}",
            )

        # Check DNC
        if lead.on_dnc_list:
            self.state.stats.dnc_violations += 1
            return ToolResult(
                call_id="",
                success=False,
                error=f"DNC VIOLATION: Lead {lead_id} is on Do Not Call list",
                data={"dnc_violation": True},
            )

        # Create call session
        call_id = generate_call_id()
        session = CallSession(
            call_id=call_id,
            lead_id=LeadID(lead_id),
            started_at=self.state.time.total_minutes(),
        )
        self.state.active_call = session
        self.state.stats.calls_today += 1
        self.state.stats.total_calls += 1

        # Advance time (1 minute to connect)
        self.state.time.advance_minutes(1, self.budget)

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
                call_id="",
                success=False,
                error="No active call. Start a call first.",
            )

        session = self.state.active_call

        # Check offer limit
        if len(session.offers_presented) >= self.budget.max_offers_per_call:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Maximum offers per call reached: {self.budget.max_offers_per_call}",
            )

        # Validate plan_id
        try:
            plan_type = PlanType(plan_id)
        except ValueError:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Invalid plan_id: {plan_id}. Valid: {[p.value for p in PlanType]}",
            )

        # Validate next_step
        try:
            next_step_enum = NextStep(next_step)
        except ValueError:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Invalid next_step: {next_step}. Valid: {[n.value for n in NextStep]}",
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
                call_id="",
                success=False,
                error=f"Lead not found: {session.lead_id}",
            )

        # Invoke buyer simulator (LLM required)
        if not self._buyer_simulator:
            return ToolResult(
                call_id="",
                success=False,
                error="Buyer simulator not configured. LLM buyer is required.",
            )

        # Get negotiation history for this lead from episode context
        negotiation_history = None
        if self._episode_context:
            negotiation_history = self._episode_context.get_buyer_view(str(session.lead_id))

        # Note: pitch is no longer passed as a tool argument - the seller's message
        # is now separate from tool calls. Pass None for backwards compatibility.
        buyer_response = self._buyer_simulator(lead, offer, session, None, negotiation_history)

        # Record offer and response
        session.offers_presented.append(offer)
        session.buyer_responses.append(buyer_response)

        # Advance time (3-5 minutes per offer presentation)
        self.state.time.advance_minutes(4, self.budget)
        session.duration_minutes += 4

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
            result_data["message"] = "Buyer ACCEPTED the plan! Call ended successfully."
            result_data["call_ended"] = True
            # Auto-end call on acceptance
            self._finalize_call(session)

        elif buyer_response.decision == BuyerDecision.REJECT_PLAN:
            self.state.record_call_outcome(BuyerDecision.REJECT_PLAN)
            result_data["message"] = "Buyer REJECTED this offer. You may present another plan."
            result_data["can_continue"] = (
                len(session.offers_presented) < self.budget.max_offers_per_call
            )

        elif buyer_response.decision == BuyerDecision.END_CALL:
            session.outcome = BuyerDecision.END_CALL
            self.state.record_call_outcome(BuyerDecision.END_CALL)
            result_data["message"] = "Buyer ENDED the call."
            result_data["call_ended"] = True

            # Check if buyer requested DNC
            if lead.hidden.dnc_risk > 0.5 and self._check_dnc_trigger(lead, session):
                lead.on_dnc_list = True
                result_data["dnc_requested"] = True
                result_data["message"] += " Lead requested Do Not Call."

            # End the call
            self._finalize_call(session)

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
            return ToolResult(
                call_id="",
                success=False,
                error="No active call to end.",
            )

        session = self.state.active_call
        self._finalize_call(session)

        lead = self.state.get_lead(session.lead_id)

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

    def _check_dnc_trigger(self, lead: "Persona", session: CallSession) -> bool:
        """Check if the buyer requests Do Not Call."""
        # DNC more likely if:
        # - Low trust
        # - Multiple offers presented
        # - Hostile temperature
        from salesbench.core.types import LeadTemperature

        if lead.temperature == LeadTemperature.HOSTILE:
            return True
        if len(session.offers_presented) >= 3 and lead.hidden.trust < 0.3:
            return True
        return False

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

        # Get conversation history
        negotiation_history = None
        if self._episode_context:
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
                    call_id="",
                    success=False,
                    error="Missing required argument: lead_id",
                )
            return self.start_call(arguments["lead_id"])

        elif method_name == "propose_plan":
            required = ["plan_id", "monthly_premium", "coverage_amount", "next_step"]
            missing = [r for r in required if r not in arguments]
            if missing:
                return ToolResult(
                    call_id="",
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
                call_id="",
                success=False,
                error=f"Unknown calling tool: {tool_name}",
            )
