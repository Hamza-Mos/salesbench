"""Scoring for SalesBench.

Score = Total Revenue (sum of monthly premiums from accepted plans).

This is:
- Interpretable ("agent earned $2,400 in monthly premiums")
- Aligned with business goal (maximize revenue)
- No config needed - revenue is revenue

DNC violations and other metrics are tracked separately for analysis.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from salesbench.core.types import BuyerDecision

if TYPE_CHECKING:
    from salesbench.envs.sales_mvp.state import EnvironmentState


@dataclass
class RevenueMetrics:
    """Revenue and metrics breakdown from an episode."""

    # Revenue (the score)
    total_revenue: float = 0.0

    # Counts for analysis
    num_accepts: int = 0
    num_rejects: int = 0
    num_end_calls: int = 0
    num_dnc_violations: int = 0

    @property
    def revenue_per_accept(self) -> float:
        """Average revenue per accepted offer."""
        if self.num_accepts == 0:
            return 0.0
        return self.total_revenue / self.num_accepts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_revenue": self.total_revenue,
            "revenue_per_accept": self.revenue_per_accept,
            "counts": {
                "num_accepts": self.num_accepts,
                "num_rejects": self.num_rejects,
                "num_end_calls": self.num_end_calls,
                "num_dnc_violations": self.num_dnc_violations,
            },
        }


def calculate_episode_revenue(state: "EnvironmentState") -> RevenueMetrics:
    """Calculate total revenue from the episode state.

    Score = sum of monthly premiums from accepted plans.

    Args:
        state: Final environment state.

    Returns:
        RevenueMetrics with revenue and counts.
    """
    metrics = RevenueMetrics()

    # Process call history for decisions
    for call in state.call_history:
        for i, response in enumerate(call.buyer_responses):
            decision = response.decision

            if decision == BuyerDecision.ACCEPT_PLAN:
                offer = call.offers_presented[i] if i < len(call.offers_presented) else None
                if offer:
                    metrics.total_revenue += offer.monthly_premium
                metrics.num_accepts += 1

            elif decision == BuyerDecision.REJECT_PLAN:
                metrics.num_rejects += 1

            elif decision == BuyerDecision.END_CALL:
                metrics.num_end_calls += 1

    # Record DNC violations (tracked separately, not a score penalty)
    metrics.num_dnc_violations = state.stats.dnc_violations

    return metrics
