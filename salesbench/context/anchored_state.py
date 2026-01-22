"""Anchored state management for conversation context.

Contains critical state that survives context compaction - injected fresh each turn.
This solves the problem of search results being lost during context compression.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


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
    # Protocol warnings (e.g., repeated propose_plan without message)
    protocol_warnings: list[str] = field(default_factory=list)

    def record_search(self, filters: dict) -> None:
        """Record that a search was performed with given filters."""
        if temp := filters.get("temperature"):
            self.searched_temperatures.add(temp)

    def record_search_results(self, leads: list[dict], turn: int) -> None:
        """Anchor leads found - these survive any compaction."""
        new_leads = 0
        for lead in leads:
            lead_id = lead.get("lead_id")
            if lead_id and lead_id not in self.accepted_lead_ids:
                if lead_id not in self.found_leads:
                    new_leads += 1
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
        if leads:
            logger.debug(
                f"[AnchoredState] Anchored {new_leads} new leads from search "
                f"(total: {len(self.found_leads)} found, {len(self.called_lead_ids)} called)"
            )

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
        logger.debug(
            f"[AnchoredState] Anchored decision: {lead_id} -> {decision} at turn {turn}"
        )
        if decision == "accept_plan":
            self.accepted_lead_ids.add(lead_id)
            self.found_leads.pop(lead_id, None)
            logger.info(
                f"[AnchoredState] Lead {lead_id} ACCEPTED - "
                f"total accepted: {len(self.accepted_lead_ids)}"
            )
        # Call ends after any decision
        self.record_call_ended(lead_id)

    def get_uncalled_leads(self) -> list[LeadSummary]:
        """Get leads that haven't been called yet."""
        return [
            l
            for l in self.found_leads.values()
            if l.lead_id not in self.called_lead_ids and l.lead_id not in self.accepted_lead_ids
        ]

    def add_protocol_warning(self, warning: str) -> None:
        """Add a protocol warning (deduplicated)."""
        if warning not in self.protocol_warnings:
            self.protocol_warnings.append(warning)

    def clear_protocol_warnings(self) -> None:
        """Clear all protocol warnings."""
        self.protocol_warnings.clear()

    def to_context_block(self) -> str:
        """Generate context block injected at start of every seller view.

        Shows facts clearly - no directive language. The model decides based on context.
        """
        lines = []

        # PROTOCOL WARNINGS - always show at top (survives context compression)
        if self.protocol_warnings:
            lines.append("!!! PROTOCOL WARNINGS !!!")
            for warning in self.protocol_warnings:
                lines.append(f"  - {warning}")
            lines.append("")

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
