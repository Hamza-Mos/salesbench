"""Scoring rubric for SalesBench.

Implements the RL-ready scoring system with:
- Primary rewards (accepts, closes)
- Profit proxy (premium-based)
- Efficiency bonuses (time, cost)
- Penalties (rejections, DNC)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from salesbench.core.config import ScoringConfig
from salesbench.core.types import BuyerDecision, NextStep

if TYPE_CHECKING:
    from salesbench.envs.sales_mvp.state import EnvironmentState


@dataclass
class ScoreComponents:
    """Breakdown of score components."""

    # Primary rewards
    accept_rewards: float = 0.0
    close_bonuses: float = 0.0
    followup_bonuses: float = 0.0

    # Profit proxy
    premium_rewards: float = 0.0

    # Efficiency
    time_efficiency_bonus: float = 0.0
    cost_efficiency_bonus: float = 0.0

    # Penalties
    reject_penalties: float = 0.0
    end_call_penalties: float = 0.0
    dnc_penalties: float = 0.0

    # Counts
    num_accepts: int = 0
    num_rejects: int = 0
    num_end_calls: int = 0
    num_dnc_violations: int = 0

    @property
    def total_score(self) -> float:
        """Calculate total score."""
        return (
            self.accept_rewards
            + self.close_bonuses
            + self.followup_bonuses
            + self.premium_rewards
            + self.time_efficiency_bonus
            + self.cost_efficiency_bonus
            + self.reject_penalties
            + self.end_call_penalties
            + self.dnc_penalties
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_score": self.total_score,
            "components": {
                "accept_rewards": self.accept_rewards,
                "close_bonuses": self.close_bonuses,
                "followup_bonuses": self.followup_bonuses,
                "premium_rewards": self.premium_rewards,
                "time_efficiency_bonus": self.time_efficiency_bonus,
                "cost_efficiency_bonus": self.cost_efficiency_bonus,
                "reject_penalties": self.reject_penalties,
                "end_call_penalties": self.end_call_penalties,
                "dnc_penalties": self.dnc_penalties,
            },
            "counts": {
                "num_accepts": self.num_accepts,
                "num_rejects": self.num_rejects,
                "num_end_calls": self.num_end_calls,
                "num_dnc_violations": self.num_dnc_violations,
            },
        }


class ScoringRubric:
    """Scoring rubric implementation."""

    def __init__(self, config: Optional[ScoringConfig] = None):
        """Initialize the scoring rubric.

        Args:
            config: Scoring configuration.
        """
        self.config = config or ScoringConfig()
        self._components = ScoreComponents()

    def reset(self) -> None:
        """Reset score components."""
        self._components = ScoreComponents()

    @property
    def components(self) -> ScoreComponents:
        """Get current score components."""
        return self._components

    @property
    def total_score(self) -> float:
        """Get current total score."""
        return self._components.total_score

    def record_accept(
        self,
        monthly_premium: float,
        next_step: NextStep,
    ) -> float:
        """Record an accepted plan.

        Args:
            monthly_premium: Monthly premium of accepted plan.
            next_step: The next step that was proposed.

        Returns:
            Score added for this event.
        """
        score_added = 0.0

        # Base accept reward
        self._components.accept_rewards += self.config.accept_reward
        self._components.num_accepts += 1
        score_added += self.config.accept_reward

        # Next step bonuses
        if next_step == NextStep.CLOSE_NOW:
            self._components.close_bonuses += self.config.close_now_bonus
            score_added += self.config.close_now_bonus
        elif next_step == NextStep.SCHEDULE_FOLLOWUP:
            self._components.followup_bonuses += self.config.schedule_followup_bonus
            score_added += self.config.schedule_followup_bonus

        # Premium-based reward
        premium_bonus = monthly_premium * self.config.premium_multiplier
        self._components.premium_rewards += premium_bonus
        score_added += premium_bonus

        return score_added

    def record_reject(self) -> float:
        """Record a rejected plan.

        Returns:
            Score added (negative) for this event.
        """
        self._components.reject_penalties += self.config.reject_penalty
        self._components.num_rejects += 1
        return self.config.reject_penalty

    def record_end_call(self) -> float:
        """Record buyer ending call.

        Returns:
            Score added (negative) for this event.
        """
        self._components.end_call_penalties += self.config.end_call_penalty
        self._components.num_end_calls += 1
        return self.config.end_call_penalty

    def record_dnc_violation(self) -> float:
        """Record a DNC violation.

        Returns:
            Score added (negative) for this event.
        """
        self._components.dnc_penalties += self.config.dnc_penalty
        self._components.num_dnc_violations += 1
        return self.config.dnc_penalty

    def calculate_efficiency_bonuses(
        self,
        days_used: int,
        total_days: int,
        tool_calls: int,
        max_tool_calls: int,
    ) -> float:
        """Calculate efficiency bonuses.

        Args:
            days_used: Days used in the episode.
            total_days: Total available days.
            tool_calls: Total tool calls made.
            max_tool_calls: Estimated max useful tool calls.

        Returns:
            Total efficiency bonus.
        """
        total_bonus = 0.0

        # Time efficiency: bonus for finishing early
        if days_used < total_days and self._components.num_accepts > 0:
            time_saved_ratio = (total_days - days_used) / total_days
            time_bonus = (
                time_saved_ratio
                * self._components.accept_rewards
                * self.config.time_efficiency_weight
            )
            self._components.time_efficiency_bonus = time_bonus
            total_bonus += time_bonus

        # Cost efficiency: bonus for fewer tool calls
        if tool_calls < max_tool_calls and self._components.num_accepts > 0:
            calls_saved_ratio = (max_tool_calls - tool_calls) / max_tool_calls
            cost_bonus = (
                calls_saved_ratio
                * self._components.accept_rewards
                * self.config.cost_efficiency_weight
            )
            self._components.cost_efficiency_bonus = cost_bonus
            total_bonus += cost_bonus

        return total_bonus

    def get_bounded_score(self) -> float:
        """Get the score bounded to configured min/max.

        Returns:
            Bounded total score.
        """
        raw_score = self._components.total_score
        return max(self.config.min_score, min(self.config.max_score, raw_score))


def calculate_episode_score(
    state: "EnvironmentState",
    config: Optional[ScoringConfig] = None,
    total_days: int = 10,
    max_tool_calls: int = 1000,
) -> ScoreComponents:
    """Calculate the final episode score from state.

    Args:
        state: Final environment state.
        config: Scoring configuration.
        total_days: Total available days.
        max_tool_calls: Estimated max useful tool calls.

    Returns:
        ScoreComponents with full breakdown.
    """
    rubric = ScoringRubric(config)

    # Process call history for decisions
    for call in state.call_history:
        for i, response in enumerate(call.buyer_responses):
            decision = response.decision

            if decision == BuyerDecision.ACCEPT_PLAN:
                offer = call.offers_presented[i] if i < len(call.offers_presented) else None
                if offer:
                    rubric.record_accept(
                        monthly_premium=offer.monthly_premium,
                        next_step=offer.next_step,
                    )

            elif decision == BuyerDecision.REJECT_PLAN:
                rubric.record_reject()

            elif decision == BuyerDecision.END_CALL:
                rubric.record_end_call()

    # Record DNC violations
    for _ in range(state.stats.dnc_violations):
        rubric.record_dnc_violation()

    # Calculate efficiency bonuses
    rubric.calculate_efficiency_bonuses(
        days_used=state.time.current_day,
        total_days=total_days,
        tool_calls=state.total_tool_calls,
        max_tool_calls=max_tool_calls,
    )

    return rubric.components
