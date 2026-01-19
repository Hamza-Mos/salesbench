"""Budget tracking for SalesBench.

Tracks and enforces:
- Time budgets (days, hours)
- Call budgets (per day, duration)
- Tool call budgets (per turn)
- Inference cost budgets
"""

from dataclasses import dataclass
from typing import Any, Optional

from salesbench.core.config import BudgetConfig
from salesbench.core.errors import BudgetExceeded


@dataclass
class BudgetUsage:
    """Tracks current budget usage."""

    # Time usage
    elapsed_days: int = 0
    elapsed_hours: int = 0
    elapsed_minutes: int = 0

    # Call usage
    calls_today: int = 0
    calls_total: int = 0
    call_minutes_total: int = 0

    # Tool usage
    tool_calls_this_turn: int = 0
    tool_calls_total: int = 0

    # Inference cost (optional)
    inference_tokens_input: int = 0
    inference_tokens_output: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": {
                "elapsed_days": self.elapsed_days,
                "elapsed_hours": self.elapsed_hours,
                "elapsed_minutes": self.elapsed_minutes,
            },
            "calls": {
                "calls_today": self.calls_today,
                "calls_total": self.calls_total,
                "call_minutes_total": self.call_minutes_total,
            },
            "tools": {
                "tool_calls_this_turn": self.tool_calls_this_turn,
                "tool_calls_total": self.tool_calls_total,
            },
            "inference": {
                "tokens_input": self.inference_tokens_input,
                "tokens_output": self.inference_tokens_output,
            },
        }


class BudgetTracker:
    """Tracks and enforces budget limits."""

    def __init__(self, config: BudgetConfig):
        """Initialize the budget tracker.

        Args:
            config: Budget configuration with limits.
        """
        self.config = config
        self.usage = BudgetUsage()

    def reset(self) -> None:
        """Reset all usage counters."""
        self.usage = BudgetUsage()

    def reset_turn(self) -> None:
        """Reset per-turn counters."""
        self.usage.tool_calls_this_turn = 0

    def reset_day(self) -> None:
        """Reset per-day counters."""
        self.usage.calls_today = 0

    def record_time(self, day: int, hour: int, minute: int) -> None:
        """Record current time.

        Args:
            day: Current day (1-10).
            hour: Current hour (9-17).
            minute: Current minute (0-59).
        """
        self.usage.elapsed_days = day - 1
        self.usage.elapsed_hours = (day - 1) * 8 + (hour - 9)
        self.usage.elapsed_minutes = self.usage.elapsed_hours * 60 + minute

    def record_call_start(self) -> None:
        """Record that a call was started."""
        self.usage.calls_today += 1
        self.usage.calls_total += 1

    def record_call_end(self, duration_minutes: int) -> None:
        """Record that a call ended.

        Args:
            duration_minutes: How long the call lasted.
        """
        self.usage.call_minutes_total += duration_minutes

    def record_tool_call(self) -> None:
        """Record a tool call."""
        self.usage.tool_calls_this_turn += 1
        self.usage.tool_calls_total += 1

    def record_inference(self, input_tokens: int, output_tokens: int) -> None:
        """Record inference token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        self.usage.inference_tokens_input += input_tokens
        self.usage.inference_tokens_output += output_tokens

    def check_time_budget(self) -> Optional[str]:
        """Check if time budget is exceeded.

        Returns:
            Error message if exceeded, None otherwise.
        """
        if self.usage.elapsed_days >= self.config.total_days:
            return f"Time budget exceeded: {self.config.total_days} days"
        return None

    def check_call_budget(self) -> Optional[str]:
        """Check if daily call budget is exceeded.

        Returns:
            Error message if exceeded, None otherwise.
        """
        if self.usage.calls_today >= self.config.max_calls_per_day:
            return f"Daily call limit reached: {self.config.max_calls_per_day}"
        return None

    def check_tool_budget(self) -> Optional[str]:
        """Check if tool call budget for this turn is exceeded.

        Returns:
            Error message if exceeded, None otherwise.
        """
        if self.usage.tool_calls_this_turn >= self.config.max_tool_calls_per_turn:
            return f"Tool call limit reached: {self.config.max_tool_calls_per_turn}"
        return None

    def check_call_duration(self, current_duration: int) -> Optional[str]:
        """Check if call duration limit is exceeded.

        Args:
            current_duration: Current call duration in minutes.

        Returns:
            Error message if exceeded, None otherwise.
        """
        if current_duration >= self.config.max_call_duration_minutes:
            return f"Call duration limit reached: {self.config.max_call_duration_minutes} minutes"
        return None

    def check_all(self) -> Optional[str]:
        """Check all budgets.

        Returns:
            First error message found, or None if all OK.
        """
        checks = [
            self.check_time_budget(),
            self.check_call_budget(),
            self.check_tool_budget(),
        ]
        for check in checks:
            if check:
                return check
        return None

    def enforce_tool_call(self) -> None:
        """Enforce tool call budget, raising if exceeded.

        Raises:
            BudgetExceeded: If tool call limit exceeded.
        """
        error = self.check_tool_budget()
        if error:
            raise BudgetExceeded(
                error,
                budget_type="tool_calls_per_turn",
                limit=self.config.max_tool_calls_per_turn,
                current=self.usage.tool_calls_this_turn,
            )

    def enforce_call_start(self) -> None:
        """Enforce call start budget, raising if exceeded.

        Raises:
            BudgetExceeded: If daily call limit exceeded.
        """
        error = self.check_call_budget()
        if error:
            raise BudgetExceeded(
                error,
                budget_type="calls_per_day",
                limit=self.config.max_calls_per_day,
                current=self.usage.calls_today,
            )

    def get_remaining(self) -> dict[str, Any]:
        """Get remaining budget amounts.

        Returns:
            Dict with remaining amounts for each budget type.
        """
        return {
            "days_remaining": self.config.total_days - self.usage.elapsed_days,
            "calls_remaining_today": self.config.max_calls_per_day - self.usage.calls_today,
            "tool_calls_remaining_this_turn": (
                self.config.max_tool_calls_per_turn - self.usage.tool_calls_this_turn
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dict with usage and remaining amounts.
        """
        return {
            "usage": self.usage.to_dict(),
            "remaining": self.get_remaining(),
            "limits": {
                "total_days": self.config.total_days,
                "max_calls_per_day": self.config.max_calls_per_day,
                "max_call_duration_minutes": self.config.max_call_duration_minutes,
                "max_tool_calls_per_turn": self.config.max_tool_calls_per_turn,
            },
        }
