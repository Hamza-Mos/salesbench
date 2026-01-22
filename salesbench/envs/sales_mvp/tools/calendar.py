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

    def get_availability(self, from_hour: Optional[int] = None, num_hours: int = 8) -> ToolResult:
        """Get available time slots.

        Args:
            from_hour: Starting hour to check. If None, returns from current elapsed hour.
            num_hours: Number of hours to check (default 8).

        Returns:
            ToolResult with available hours.
        """
        if from_hour is None:
            from_hour = self.state.time.elapsed_hours

        if from_hour < 0 or from_hour >= self.budget.total_hours:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Invalid hour: {from_hour}. Must be 0-{self.budget.total_hours - 1}",
            )

        if from_hour < self.state.time.elapsed_hours:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Cannot check availability for past hour {from_hour}",
            )

        available_hours = self.state.get_availability(
            from_hour, num_hours, max_hour=self.budget.total_hours
        )

        # Filter to only future hours (at least 1 hour ahead)
        min_hour = self.state.time.elapsed_hours + 1
        available_hours = [h for h in available_hours if h >= min_hour]

        # Filter to only hours within budget
        available_hours = [h for h in available_hours if h < self.budget.total_hours]

        return ToolResult(
            call_id="",
            success=True,
            data={
                "from_hour": from_hour,
                "available_hours": available_hours,
                "current_time": {
                    "elapsed_hours": self.state.time.elapsed_hours,
                    "elapsed_minutes": self.state.time.elapsed_minutes,
                },
                "total_hours": self.budget.total_hours,
            },
        )

    def schedule_call(
        self,
        lead_id: str,
        hour: int,
    ) -> ToolResult:
        """Schedule a call with a lead.

        Args:
            lead_id: The lead to schedule with.
            hour: Hour to schedule (elapsed hour, must be in future).

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

        # Validate hour is in valid range
        if hour < 0 or hour >= self.budget.total_hours:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Invalid hour: {hour}. Must be 0-{self.budget.total_hours - 1}",
            )

        # Must be at least 1 hour in the future
        if hour <= self.state.time.elapsed_hours:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Cannot schedule in the past or current hour (hour {hour}, current is {self.state.time.elapsed_hours})",
            )

        # Try to schedule
        appointment = self.state.schedule_appointment(
            lead_id=LeadID(lead_id),
            scheduled_hour=hour,
        )

        if not appointment:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Time slot unavailable: Hour {hour}",
            )

        return ToolResult(
            call_id="",
            success=True,
            data={
                "appointment_id": appointment.appointment_id,
                "lead_id": lead_id,
                "lead_name": lead.name,
                "scheduled_hour": hour,
                "scheduled_time": f"Hour {hour}",
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
            return self.get_availability(
                from_hour=arguments.get("from_hour"),
                num_hours=arguments.get("num_hours", 8),
            )
        elif method_name == "schedule_call":
            required = ["lead_id", "hour"]
            missing = [r for r in required if r not in arguments]
            if missing:
                return ToolResult(
                    call_id="",
                    success=False,
                    error=f"Missing required arguments: {missing}",
                )
            return self.schedule_call(
                lead_id=arguments["lead_id"],
                hour=arguments["hour"],
            )
        else:
            return ToolResult(
                call_id="",
                success=False,
                error=f"Unknown calendar tool: {tool_name}",
            )
