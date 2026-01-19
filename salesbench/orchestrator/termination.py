"""Termination condition checking for SalesBench.

Checks for:
- Time-based termination (10 days elapsed)
- Budget exhaustion (no more calls available)
- Lead exhaustion (all leads on DNC or sold)
- Agent-requested termination
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from salesbench.core.config import BudgetConfig
    from salesbench.envs.sales_mvp.state import EnvironmentState


class TerminationReason(str, Enum):
    """Reasons for episode termination."""

    TIME_LIMIT = "time_limit"  # 10 days elapsed
    NO_LEADS = "no_leads"  # All leads exhausted
    BUDGET_EXHAUSTED = "budget_exhausted"  # No more calls/actions available
    AGENT_QUIT = "agent_quit"  # Agent chose to end
    MAX_TURNS = "max_turns"  # Maximum turns reached
    ERROR = "error"  # Unrecoverable error


@dataclass
class TerminationStatus:
    """Status of termination check."""

    terminated: bool
    reason: Optional[TerminationReason] = None
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "terminated": self.terminated,
            "reason": self.reason.value if self.reason else None,
            "message": self.message,
        }


class TerminationChecker:
    """Checks termination conditions."""

    def __init__(self, budget: "BudgetConfig", max_turns: Optional[int] = None):
        """Initialize the termination checker.

        Args:
            budget: Budget configuration.
            max_turns: Maximum number of turns (optional).
        """
        self.budget = budget
        self.max_turns = max_turns
        self._turn_count = 0

    def reset(self) -> None:
        """Reset termination state."""
        self._turn_count = 0

    def increment_turn(self) -> None:
        """Increment the turn counter."""
        self._turn_count += 1

    @property
    def turn_count(self) -> int:
        """Get current turn count."""
        return self._turn_count

    def check_time_termination(self, state: "EnvironmentState") -> TerminationStatus:
        """Check if time limit has been reached.

        Args:
            state: Current environment state.

        Returns:
            TerminationStatus indicating if terminated.
        """
        if state.time.is_episode_ended(self.budget):
            return TerminationStatus(
                terminated=True,
                reason=TerminationReason.TIME_LIMIT,
                message=f"Episode ended: {self.budget.total_days} business days elapsed",
            )
        return TerminationStatus(terminated=False)

    def check_lead_exhaustion(self, state: "EnvironmentState") -> TerminationStatus:
        """Check if all leads are exhausted.

        Args:
            state: Current environment state.

        Returns:
            TerminationStatus indicating if terminated.
        """
        active_leads = sum(1 for lead in state.leads.values() if not lead.on_dnc_list)

        if active_leads == 0:
            return TerminationStatus(
                terminated=True,
                reason=TerminationReason.NO_LEADS,
                message="No more leads available (all on DNC list)",
            )
        return TerminationStatus(terminated=False)

    def check_turn_limit(self) -> TerminationStatus:
        """Check if maximum turns reached.

        Returns:
            TerminationStatus indicating if terminated.
        """
        if self.max_turns and self._turn_count >= self.max_turns:
            return TerminationStatus(
                terminated=True,
                reason=TerminationReason.MAX_TURNS,
                message=f"Maximum turns reached: {self.max_turns}",
            )
        return TerminationStatus(terminated=False)

    def check_all(self, state: "EnvironmentState") -> TerminationStatus:
        """Check all termination conditions.

        Args:
            state: Current environment state.

        Returns:
            First termination status found, or not terminated.
        """
        checks = [
            self.check_time_termination(state),
            self.check_lead_exhaustion(state),
            self.check_turn_limit(),
        ]

        for status in checks:
            if status.terminated:
                return status

        return TerminationStatus(terminated=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_count": self._turn_count,
            "max_turns": self.max_turns,
            "budget_total_days": self.budget.total_days,
        }
