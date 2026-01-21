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
    """Tracks simulated time."""

    current_day: int = 1  # Days 1-10
    current_hour: int = 9  # Hours 9-17 (9 AM - 5 PM)
    current_minute: int = 0  # Minutes 0-59

    def total_minutes(self) -> int:
        """Get total elapsed minutes from start."""
        day_minutes = (self.current_day - 1) * 8 * 60
        hour_minutes = (self.current_hour - 9) * 60
        return day_minutes + hour_minutes + self.current_minute

    def advance_minutes(self, minutes: int, budget: BudgetConfig) -> None:
        """Advance time by given minutes."""
        self.current_minute += minutes

        while self.current_minute >= 60:
            self.current_minute -= 60
            self.current_hour += 1

        while self.current_hour >= 17:  # End of day
            self.current_hour = 9  # Start of next day
            self.current_day += 1

    def is_end_of_day(self) -> bool:
        """Check if it's end of business day."""
        return self.current_hour >= 17

    def is_episode_ended(self, budget: BudgetConfig) -> bool:
        """Check if episode time has ended."""
        return self.current_day > budget.total_days

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_day": self.current_day,
            "current_hour": self.current_hour,
            "current_minute": self.current_minute,
            "total_minutes": self.total_minutes(),
        }


@dataclass
class CallStats:
    """Statistics for calls made."""

    calls_today: int = 0
    total_calls: int = 0
    total_call_minutes: int = 0
    accepted_offers: int = 0
    rejected_offers: int = 0
    calls_ended_by_buyer: int = 0
    dnc_violations: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "calls_today": self.calls_today,
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

    def reset_day(self) -> None:
        """Reset per-day counters."""
        self.stats.calls_today = 0

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
        day: int,
        hour: int,
    ) -> Optional[Appointment]:
        """Schedule an appointment."""
        if lead_id not in self.leads:
            return None

        # Check for conflicts
        for appt in self.appointments.values():
            if appt.scheduled_day == day and appt.scheduled_hour == hour:
                return None

        appointment = Appointment(
            appointment_id=generate_appointment_id(),
            lead_id=lead_id,
            scheduled_day=day,
            scheduled_hour=hour,
            created_at=self.time.total_minutes(),
        )
        self.appointments[appointment.appointment_id] = appointment
        return appointment

    def get_availability(self, day: int) -> list[int]:
        """Get available hours for a given day."""
        booked = set()
        for appt in self.appointments.values():
            if appt.scheduled_day == day and not appt.completed:
                booked.add(appt.scheduled_hour)
        return [h for h in range(9, 17) if h not in booked]

    def get_scheduled_for_now(self) -> list[Appointment]:
        """Get appointments scheduled for current time."""
        return [
            appt
            for appt in self.appointments.values()
            if appt.scheduled_day == self.time.current_day
            and appt.scheduled_hour == self.time.current_hour
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
