"""Environment state management for SalesBench.

Manages:
- Leads/CRM data
- Calendar/appointments
- Active call sessions
- Time tracking
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from salesbench.core.config import BudgetConfig
from salesbench.core.types import (
    Appointment,
    AppointmentID,
    BuyerDecision,
    CallSession,
    LeadID,
    generate_appointment_id,
)
from salesbench.envs.sales_mvp.personas import Persona


@dataclass
class TimeState:
    """Tracks simulated time as elapsed hours and minutes."""

    elapsed_hours: int = 0  # Total hours elapsed (starts at 0)
    elapsed_minutes: int = 0  # Minutes within current hour (0-59)

    def total_minutes(self) -> int:
        """Get total elapsed minutes from start."""
        return self.elapsed_hours * 60 + self.elapsed_minutes

    def advance_minutes(self, minutes: int, budget: BudgetConfig) -> None:
        """Advance time by given minutes.

        Args:
            minutes: Number of minutes to advance.
            budget: Budget config (unused, kept for API compatibility).
        """
        self.elapsed_minutes += minutes

        while self.elapsed_minutes >= 60:
            self.elapsed_minutes -= 60
            self.elapsed_hours += 1

    def is_episode_ended(self, budget: BudgetConfig) -> bool:
        """Check if episode time has ended."""
        return self.elapsed_hours >= budget.total_hours

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "elapsed_hours": self.elapsed_hours,
            "elapsed_minutes": self.elapsed_minutes,
            "total_minutes": self.total_minutes(),
        }


@dataclass
class CallStats:
    """Statistics for calls made."""

    total_calls: int = 0
    total_call_minutes: int = 0
    accepted_offers: int = 0
    rejected_offers: int = 0
    calls_ended_by_buyer: int = 0
    dnc_violations: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_calls": self.total_calls,
            "total_call_minutes": self.total_call_minutes,
            "accepted_offers": self.accepted_offers,
            "rejected_offers": self.rejected_offers,
            "calls_ended_by_buyer": self.calls_ended_by_buyer,
            "dnc_violations": self.dnc_violations,
        }


@dataclass
class EnvironmentState:
    """Complete environment state."""

    # Leads/CRM
    leads: dict[LeadID, Persona] = field(default_factory=dict)

    # Calendar
    appointments: dict[AppointmentID, Appointment] = field(default_factory=dict)

    # Call sessions
    call_history: list[CallSession] = field(default_factory=list)
    active_call: Optional[CallSession] = None

    # Time tracking
    time: TimeState = field(default_factory=TimeState)

    # Statistics
    stats: CallStats = field(default_factory=CallStats)

    # Tool call tracking
    tool_calls_this_turn: int = 0
    total_tool_calls: int = 0

    def reset_turn(self) -> None:
        """Reset per-turn counters."""
        self.tool_calls_this_turn = 0

    def get_lead(self, lead_id: LeadID) -> Optional[Persona]:
        """Get a lead by ID."""
        return self.leads.get(lead_id)

    def search_leads(
        self,
        temperature: Optional[str] = None,
        min_income: Optional[int] = None,
        max_age: Optional[int] = None,
        limit: int = 10,
    ) -> list[Persona]:
        """Search leads with filters."""
        results = []
        for lead in self.leads.values():
            if lead.on_dnc_list:
                continue
            if lead.converted:
                continue
            if temperature and lead.temperature.value != temperature:
                continue
            if min_income and lead.annual_income < min_income:
                continue
            if max_age and lead.age > max_age:
                continue
            results.append(lead)
            if len(results) >= limit:
                break
        return results

    def schedule_appointment(
        self,
        lead_id: LeadID,
        scheduled_hour: int,
    ) -> Optional[Appointment]:
        """Schedule an appointment at a future hour.

        Args:
            lead_id: The lead to schedule.
            scheduled_hour: The elapsed hour to schedule at (must be >= current elapsed_hours).
        """
        if lead_id not in self.leads:
            return None

        # Check for conflicts
        for appt in self.appointments.values():
            if appt.scheduled_hour == scheduled_hour:
                return None

        appointment = Appointment(
            appointment_id=generate_appointment_id(),
            lead_id=lead_id,
            scheduled_day=1,  # Kept for compatibility, not used
            scheduled_hour=scheduled_hour,
            created_at=self.time.total_minutes(),
        )
        self.appointments[appointment.appointment_id] = appointment
        return appointment

    def get_availability(
        self, from_hour: int, num_hours: int = 8, max_hour: Optional[int] = None
    ) -> list[int]:
        """Get available hours starting from a given hour.

        Args:
            from_hour: Starting hour to check.
            num_hours: Number of hours to check (default 8).
            max_hour: Maximum valid hour (exclusive). If provided, caps the range.
        """
        booked = set()
        for appt in self.appointments.values():
            if not appt.completed:
                booked.add(appt.scheduled_hour)

        end_hour = from_hour + num_hours
        if max_hour is not None:
            end_hour = min(end_hour, max_hour)

        return [h for h in range(from_hour, end_hour) if h not in booked]

    def get_scheduled_for_now(self) -> list[Appointment]:
        """Get appointments scheduled for current time."""
        return [
            appt
            for appt in self.appointments.values()
            if appt.scheduled_hour == self.time.elapsed_hours
            and not appt.completed
        ]

    def record_call_outcome(self, outcome: BuyerDecision) -> None:
        """Record the outcome of a call."""
        if outcome == BuyerDecision.ACCEPT_PLAN:
            self.stats.accepted_offers += 1
        elif outcome == BuyerDecision.REJECT_PLAN:
            self.stats.rejected_offers += 1
        elif outcome == BuyerDecision.END_CALL:
            self.stats.calls_ended_by_buyer += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "time": self.time.to_dict(),
            "stats": self.stats.to_dict(),
            "leads_count": len(self.leads),
            "appointments_count": len(self.appointments),
            "call_history_count": len(self.call_history),
            "has_active_call": self.active_call is not None,
            "tool_calls_this_turn": self.tool_calls_this_turn,
            "total_tool_calls": self.total_tool_calls,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Convert to full dictionary including all data."""
        return {
            "time": self.time.to_dict(),
            "stats": self.stats.to_dict(),
            "leads": {lid: lead.to_public_dict() for lid, lead in self.leads.items()},
            "appointments": {aid: appt.to_dict() for aid, appt in self.appointments.items()},
            "call_history": [call.to_dict() for call in self.call_history],
            "active_call": self.active_call.to_dict() if self.active_call else None,
            "tool_calls_this_turn": self.tool_calls_this_turn,
            "total_tool_calls": self.total_tool_calls,
        }
