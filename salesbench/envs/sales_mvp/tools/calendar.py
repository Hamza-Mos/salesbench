"""Calendar tools for the sales environment.

Tools:
- calendar.get_availability: Get available time slots
- calendar.schedule_call: Schedule a call with a lead
"""

from typing import TYPE_CHECKING, Any, Optional

from salesbench.core.config import BudgetConfig
from salesbench.core.types import LeadID, ToolResult

if TYPE_CHECKING:
    from salesbench.envs.sales_mvp.state import EnvironmentState


class CalendarTools:
    """Calendar tool implementations."""

    def __init__(self, state: "EnvironmentState", budget: BudgetConfig):
        self.state = state
        self.budget = budget

    def get_availability(self, day: Optional[int] = None) -> ToolResult:
        """Get available time slots.

        Args:
            day: Specific day to check (1-10). If None, returns current day.

        Returns:
            ToolResult with available hours.
        """
        if day is None:
            day = self.state.time.current_day

        if day < 1 or day > self.budget.total_days:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Invalid day: {day}. Must be 1-{self.budget.total_days}",
            )

        if day < self.state.time.current_day:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Cannot check availability for past day {day}",
            )

        available_hours = self.state.get_availability(day)

        # If checking current day, filter out past hours
        if day == self.state.time.current_day:
            available_hours = [h for h in available_hours if h > self.state.time.current_hour]

        return ToolResult(
            call_id="",
            success=True,
            data={
                "day": day,
                "available_hours": available_hours,
                "is_today": day == self.state.time.current_day,
                "current_time": {
                    "day": self.state.time.current_day,
                    "hour": self.state.time.current_hour,
                    "minute": self.state.time.current_minute,
                },
            },
        )

    def schedule_call(
        self,
        lead_id: str,
        day: int,
        hour: int,
    ) -> ToolResult:
        """Schedule a call with a lead.

        Args:
            lead_id: The lead to schedule with.
            day: Day to schedule (1-10).
            hour: Hour to schedule (9-17).

        Returns:
            ToolResult with appointment details.
        """
        # Validate lead
        lead = self.state.get_lead(LeadID(lead_id))
        if not lead:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Lead not found: {lead_id}",
            )

        if lead.on_dnc_list:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Cannot schedule call with DNC lead: {lead_id}",
            )

        # Validate day
        if day < 1 or day > self.budget.total_days:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Invalid day: {day}. Must be 1-{self.budget.total_days}",
            )

        if day < self.state.time.current_day:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Cannot schedule in the past (day {day})",
            )

        # Validate hour
        if hour < 9 or hour > 16:  # Can't schedule at 5 PM, no time for call
            return ToolResult(
                call_id="",
                success=False,
                error=f"Invalid hour: {hour}. Must be 9-16 (9 AM - 4 PM)",
            )

        # If same day, check it's in the future
        if day == self.state.time.current_day and hour <= self.state.time.current_hour:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Cannot schedule in the past (hour {hour})",
            )

        # Try to schedule
        appointment = self.state.schedule_appointment(
            lead_id=LeadID(lead_id),
            day=day,
            hour=hour,
        )

        if not appointment:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Time slot unavailable: Day {day}, Hour {hour}",
            )

        return ToolResult(
            call_id="",
            success=True,
            data={
                "appointment_id": appointment.appointment_id,
                "lead_id": lead_id,
                "lead_name": lead.name,
                "scheduled_day": day,
                "scheduled_hour": hour,
                "scheduled_time": f"Day {day}, {hour}:00",
            },
        )

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a calendar tool.

        Args:
            tool_name: Full tool name (e.g., "calendar.get_availability").
            arguments: Tool arguments.

        Returns:
            ToolResult from the tool execution.
        """
        method_name = tool_name.replace("calendar.", "")

        if method_name == "get_availability":
            return self.get_availability(day=arguments.get("day"))
        elif method_name == "schedule_call":
            required = ["lead_id", "day", "hour"]
            missing = [r for r in required if r not in arguments]
            if missing:
                return ToolResult(
                    call_id="",
                    success=False,
                    error=f"Missing required arguments: {missing}",
                )
            return self.schedule_call(
                lead_id=arguments["lead_id"],
                day=arguments["day"],
                hour=arguments["hour"],
            )
        else:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Unknown calendar tool: {tool_name}",
            )
