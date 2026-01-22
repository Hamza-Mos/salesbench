"""Heuristic seller agent for baseline comparison.

This is a deterministic, rule-based seller agent that:
1. Searches for HOT/WARM leads first
2. Calls leads in priority order
3. Proposes TERM plan with standard coverage
4. Ends call after rejection
5. Never violates DNC

Used for:
- Zero-cost debugging (no API calls)
- Performance floor measurement
- Deterministic test fixture
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from salesbench.agents.seller_base import SellerAgent, SellerConfig, SellerObservation
from salesbench.core.protocol import SellerAction
from salesbench.core.types import (
    ToolCall,
    ToolResult,
)


@dataclass
class HeuristicState:
    """State for the heuristic seller."""

    leads_searched: bool = False
    current_lead_id: Optional[str] = None
    call_active: bool = False
    offer_made: bool = False
    leads_queue: list[dict] = field(default_factory=list)
    called_leads: set = field(default_factory=set)


class HeuristicSeller(SellerAgent):
    """A deterministic, rule-based seller agent.

    Strategy:
    1. Search for leads, prioritizing HOT > WARM > LUKEWARM
    2. For each lead, start a call
    3. Propose a TERM plan with coverage based on income
    4. If accepted, schedule followup or close
    5. If rejected, end call and move to next lead
    6. Never call DNC leads
    """

    def __init__(self, max_offers_per_call: int = 2, config: Optional[SellerConfig] = None):
        """Initialize the heuristic seller.

        Args:
            max_offers_per_call: Maximum offers to make before giving up.
            config: Optional seller configuration.
        """
        super().__init__(config)
        self.max_offers_per_call = max_offers_per_call
        self._state = HeuristicState()

    def reset(self) -> None:
        """Reset the agent state."""
        self._state = HeuristicState()
        self._total_api_cost = 0.0
        self._turn_count = 0

    def act(self, observation: SellerObservation) -> SellerAction:
        """Decide which tools to call based on observation.

        Args:
            observation: Current observation from environment.

        Returns:
            SellerAction containing tool calls.
        """
        # Update state from observation
        self._state.call_active = observation.in_call
        self._state.current_lead_id = observation.current_lead_id

        # Process previous results
        if observation.last_tool_results:
            self._process_results(observation.last_tool_results)

        tool_calls = self.generate_tool_calls(observation)
        return self._create_action(tool_calls)

    def generate_tool_calls(
        self,
        observation: Any,
        tool_results: Optional[list[ToolResult]] = None,
    ) -> list[ToolCall]:
        """Generate tool calls based on current state.

        Args:
            observation: Current environment observation.
            tool_results: Results from previous tool calls.

        Returns:
            List of tool calls to execute.
        """
        # Process previous results
        if tool_results:
            self._process_results(tool_results)

        # Decision tree
        if not self._state.leads_searched:
            return self._search_leads()

        if not self._state.leads_queue and not self._state.call_active:
            # No more leads to call
            return self._search_leads()

        if not self._state.call_active:
            return self._start_next_call()

        if not self._state.offer_made:
            return self._propose_plan(observation)

        # If we're here, offer was made and call is still active
        # End the call and move on
        return self._end_call()

    def _process_results(self, results: list[ToolResult]) -> None:
        """Process tool results to update state."""
        for result in results:
            if not result.success:
                continue

            data = result.data or {}

            # Handle search results
            if "leads" in data:
                self._state.leads_searched = True
                self._prioritize_leads(data["leads"])

            # Handle call start
            if "call_id" in data and not data.get("call_started"):
                self._state.call_active = True

            # Handle offer response
            if "decision" in data:
                decision = data["decision"]
                if decision == "accept_plan":
                    # Success! End call gracefully
                    self._state.call_active = True
                    self._state.offer_made = True
                elif decision == "reject_plan":
                    # Rejection - we'll end the call
                    self._state.offer_made = True
                elif decision == "end_call":
                    # Buyer ended - call is over
                    self._state.call_active = False
                    self._state.offer_made = False
                    self._state.current_lead_id = None

            # Handle call end
            if data.get("call_ended") or "ended" in str(data).lower():
                self._state.call_active = False
                self._state.offer_made = False
                self._state.current_lead_id = None

    def _prioritize_leads(self, leads: list[dict]) -> None:
        """Prioritize leads by temperature."""
        priority = {
            "hot": 0,
            "warm": 1,
            "lukewarm": 2,
            "cold": 3,
            "hostile": 4,
        }

        # Filter out already called and DNC leads
        available = [
            lead
            for lead in leads
            if lead["lead_id"] not in self._state.called_leads
            and not lead.get("on_dnc_list", False)
        ]

        # Sort by priority
        available.sort(key=lambda l: priority.get(l.get("temperature", "cold"), 3))

        self._state.leads_queue = available

    def _search_leads(self) -> list[ToolCall]:
        """Search for leads."""
        return [
            ToolCall(
                tool_name="crm.search_leads",
                arguments={"limit": 20},
            )
        ]

    def _start_next_call(self) -> list[ToolCall]:
        """Start a call with the next lead in queue."""
        if not self._state.leads_queue:
            # No leads left, search again
            self._state.leads_searched = False
            return self._search_leads()

        lead = self._state.leads_queue.pop(0)
        lead_id = lead["lead_id"]

        self._state.current_lead_id = lead_id
        self._state.called_leads.add(lead_id)
        self._state.offer_made = False

        return [
            ToolCall(
                tool_name="calling.start_call",
                arguments={"lead_id": lead_id},
            )
        ]

    def _propose_plan(self, observation: dict[str, Any]) -> list[ToolCall]:
        """Propose a plan to the current lead."""
        # Use standard TERM coverage based on general guidelines
        # In a real scenario, we'd look up the lead's income
        coverage = 500000  # Default to $500K

        return [
            ToolCall(
                tool_name="calling.propose_plan",
                arguments={
                    "plan_id": "TERM",
                    "coverage_amount": coverage,
                    "term_years": 20,
                    "next_step": "close_now",
                },
            )
        ]

    def _end_call(self) -> list[ToolCall]:
        """End the current call."""
        self._state.offer_made = False
        return [
            ToolCall(
                tool_name="calling.end_call",
                arguments={},
            )
        ]


def create_heuristic_seller(max_offers_per_call: int = 2) -> HeuristicSeller:
    """Create a heuristic seller agent.

    Args:
        max_offers_per_call: Maximum offers to make per call.

    Returns:
        HeuristicSeller instance.
    """
    return HeuristicSeller(max_offers_per_call=max_offers_per_call)
