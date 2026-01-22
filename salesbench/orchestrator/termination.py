"""Termination condition checking for SalesBench.

Natural termination conditions (no artificial turn limits):
- TIME_LIMIT: Total hours elapsed
- NO_LEADS: All leads resolved (converted or DNC)
- SAFETY_LIMIT: Configurable ceiling (None by default)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from salesbench.core.types import LeadStatus

if TYPE_CHECKING:
    from salesbench.core.config import BudgetConfig
    from salesbench.envs.sales_mvp.state import EnvironmentState

logger = logging.getLogger(__name__)


class TerminationReason(str, Enum):
    """Reasons for episode termination.

    Ordered by desirability - lead exhaustion is most natural,
    then time limit, with safety limit as a fallback.
    """

    NO_LEADS = "no_leads"  # All leads resolved (natural completion)
    TIME_LIMIT = "time_limit"  # 10 business days elapsed
    SAFETY_LIMIT = "safety_limit"  # Configurable ceiling hit (fallback)


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
    """Checks natural termination conditions.

    Priority order for termination checks:
    1. Lead exhaustion (natural completion - all leads resolved)
    2. Time limit (business period ended - 10 days)
    3. Safety limit (only if configured - fallback ceiling)
    """

    def __init__(self, budget: "BudgetConfig", safety_max_turns: Optional[int] = None):
        """Initialize the termination checker.

        Args:
            budget: Budget configuration.
            safety_max_turns: Optional safety ceiling for turns (None = no limit).
        """
        self.budget = budget
        self.safety_max_turns = safety_max_turns
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
                message=f"Episode ended: {self.budget.total_hours} hours elapsed",
            )
        return TerminationStatus(terminated=False)

    def check_minimum_viable_time(self, state: "EnvironmentState") -> TerminationStatus:
        """Terminate if remaining time is below minimum for any productive action.

        When there's not enough time to start a call and make at least one proposal,
        and no call is currently active, auto-terminate to prevent looping behavior
        where the seller keeps deciding not to act.

        Args:
            state: Current environment state.

        Returns:
            TerminationStatus indicating if terminated due to insufficient time.
        """
        total_minutes = self.budget.total_hours * 60
        elapsed_minutes = state.time.elapsed_hours * 60 + state.time.elapsed_minutes
        remaining_minutes = total_minutes - elapsed_minutes

        # Minimum time needed: start a call + make one proposal
        min_viable = self.budget.start_call_cost + self.budget.propose_plan_cost

        if remaining_minutes <= min_viable and state.active_call is None:
            return TerminationStatus(
                terminated=True,
                reason=TerminationReason.TIME_LIMIT,
                message=f"Insufficient time for productive action ({remaining_minutes:.1f}m <= {min_viable:.1f}m minimum)",
            )
        return TerminationStatus(terminated=False)

    def check_lead_exhaustion(self, state: "EnvironmentState") -> TerminationStatus:
        """Check if all leads are exhausted (converted or DNC).

        Args:
            state: Current environment state.

        Returns:
            TerminationStatus indicating if terminated.
        """
        # Count leads that are still active
        active_leads = sum(
            1 for lead in state.leads.values()
            if lead.status == LeadStatus.ACTIVE
        )

        if active_leads == 0:
            # Count each status for the message
            converted = sum(1 for l in state.leads.values() if l.status == LeadStatus.CONVERTED)
            dnc = sum(1 for l in state.leads.values() if l.status == LeadStatus.DNC)

            return TerminationStatus(
                terminated=True,
                reason=TerminationReason.NO_LEADS,
                message=f"All leads resolved: {converted} converted, {dnc} DNC",
            )
        return TerminationStatus(terminated=False)

    def check_safety_limit(self) -> TerminationStatus:
        """Check if safety turn limit reached.

        Returns:
            TerminationStatus indicating if safety limit hit.
        """
        if self.safety_max_turns and self._turn_count >= self.safety_max_turns:
            return TerminationStatus(
                terminated=True,
                reason=TerminationReason.SAFETY_LIMIT,
                message=f"Safety limit reached: {self.safety_max_turns} turns",
            )
        return TerminationStatus(terminated=False)

    def check_all(self, state: "EnvironmentState") -> TerminationStatus:
        """Check all termination conditions in priority order.

        Priority:
        1. Lead exhaustion (natural completion)
        2. Minimum viable time (insufficient time for productive action)
        3. Time limit (business period ended)
        4. Safety limit (only if configured)

        Args:
            state: Current environment state.

        Returns:
            First termination status found, or not terminated.
        """
        checks = [
            self.check_lead_exhaustion(state),
            self.check_minimum_viable_time(state),
            self.check_time_termination(state),
            self.check_safety_limit(),
        ]

        for status in checks:
            if status.terminated:
                self._log_termination(status)
                return status

        return TerminationStatus(terminated=False)

    def _log_termination(self, status: TerminationStatus) -> None:
        """Log termination with a clear prefix.

        Args:
            status: The termination status to log.
        """
        if status.reason:
            prefix = f"[TERMINATION:{status.reason.value.upper()}]"
            logger.info(f"{prefix} {status.message}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_count": self._turn_count,
            "safety_max_turns": self.safety_max_turns,
            "budget_total_hours": self.budget.total_hours,
        }
