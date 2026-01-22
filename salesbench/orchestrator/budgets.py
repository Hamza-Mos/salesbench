"""Budget tracking for SalesBench.

Tracks time usage - the only natural constraint in the simulation.
"""

from dataclasses import dataclass
from typing import Any

from salesbench.core.config import BudgetConfig


@dataclass
class BudgetUsage:
    """Tracks current usage."""

    # Time usage
    elapsed_hours: int = 0
    elapsed_minutes: int = 0  # Minutes within current hour (0-59)
    total_elapsed_minutes: int = 0  # Total minutes elapsed

    # Call stats (for metrics, not limits)
    calls_total: int = 0
    call_minutes_total: int = 0

    # Tool stats (for metrics, not limits)
    tool_calls_this_turn: int = 0
    tool_calls_total: int = 0

    # Inference cost tracking
    inference_tokens_input: int = 0
    inference_tokens_output: int = 0

    # Dual time metrics (always tracked regardless of time_model)
    action_based_minutes: float = 0.0  # Time from action costs
    token_based_minutes: float = 0.0  # Time estimated from tokens

    # Conversation turn tracking
    conversation_turns: int = 0  # Total conversation turns during calls

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": {
                "elapsed_hours": self.elapsed_hours,
                "elapsed_minutes": self.elapsed_minutes,
                "total_elapsed_minutes": self.total_elapsed_minutes,
            },
            "calls": {
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
            "time_metrics": {
                "action_based_minutes": self.action_based_minutes,
                "token_based_minutes": self.token_based_minutes,
                "conversation_turns": self.conversation_turns,
            },
        }


class BudgetTracker:
    """Tracks time and usage metrics."""

    def __init__(self, config: BudgetConfig):
        """Initialize the budget tracker.

        Args:
            config: Budget configuration (time settings).
        """
        self.config = config
        self.usage = BudgetUsage()

    def reset(self) -> None:
        """Reset all usage counters."""
        self.usage = BudgetUsage()

    def reset_turn(self) -> None:
        """Reset per-turn counters."""
        self.usage.tool_calls_this_turn = 0

    def record_time(self, elapsed_hours: int, elapsed_minutes: int) -> None:
        """Record current time.

        Args:
            elapsed_hours: Total elapsed hours.
            elapsed_minutes: Minutes within current hour (0-59).
        """
        self.usage.elapsed_hours = elapsed_hours
        self.usage.elapsed_minutes = elapsed_minutes
        self.usage.total_elapsed_minutes = elapsed_hours * 60 + elapsed_minutes

    def record_call_start(self) -> None:
        """Record that a call was started."""
        self.usage.calls_total += 1

    def record_call_end(self, duration_minutes: int) -> None:
        """Record that a call ended."""
        self.usage.call_minutes_total += duration_minutes

    def record_tool_call(self) -> None:
        """Record a tool call."""
        self.usage.tool_calls_this_turn += 1
        self.usage.tool_calls_total += 1

    def record_inference(self, input_tokens: int, output_tokens: int) -> None:
        """Record inference token usage."""
        self.usage.inference_tokens_input += input_tokens
        self.usage.inference_tokens_output += output_tokens

    def record_action_time(self, minutes: float) -> None:
        """Record time for an action (action-based time model).

        Args:
            minutes: Time cost in minutes for the action.
        """
        self.usage.action_based_minutes += minutes

    def record_token_time(self, tokens: int) -> None:
        """Record time based on token usage (token-based time model).

        Args:
            tokens: Number of tokens used.
        """
        self.usage.token_based_minutes += tokens / self.config.tokens_per_minute

    def record_conversation_turn(self, tokens: int = 0) -> None:
        """Record a conversation turn during an active call.

        This tracks time cost for the conversation exchange.

        Args:
            tokens: Number of tokens in this turn (for token-based tracking).
        """
        self.usage.conversation_turns += 1
        # Action-based: fixed cost per turn
        self.usage.action_based_minutes += self.config.conversation_turn_cost
        # Token-based: cost based on token count
        if tokens > 0:
            self.usage.token_based_minutes += tokens / self.config.tokens_per_minute

    def get_budget_minutes(self) -> float:
        """Get the budget minutes based on the active time model.

        Returns:
            The minutes used according to the configured time model.
        """
        if self.config.time_model == "token":
            return self.usage.token_based_minutes
        return self.usage.action_based_minutes

    def is_time_exceeded(self) -> bool:
        """Check if time budget is exceeded."""
        return self.usage.elapsed_hours >= self.config.total_hours

    def get_remaining_hours(self) -> int:
        """Get remaining hours."""
        return max(0, self.config.total_hours - self.usage.elapsed_hours)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "usage": self.usage.to_dict(),
            "total_hours": self.config.total_hours,
            "hours_remaining": self.get_remaining_hours(),
            "time_model": self.config.time_model,
            "budget_minutes_used": self.get_budget_minutes(),
        }
