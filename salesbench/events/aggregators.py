"""Event aggregators for metrics computation.

Aggregators process event streams to compute:
- Episode-level metrics
- Call-level metrics
- Performance statistics
- pass^k metrics
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from salesbench.events.event_log import Event, EventSubscriber, EventType


class EventAggregator(EventSubscriber, ABC):
    """Abstract base class for event aggregators."""

    @abstractmethod
    def aggregate(self) -> dict[str, Any]:
        """Compute aggregated metrics.

        Returns:
            Dictionary of aggregated metrics.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset aggregator state."""
        pass


@dataclass
class EpisodeAggregator(EventAggregator):
    """Aggregates episode-level metrics."""

    # Counters
    total_calls: int = 0
    total_offers: int = 0
    total_accepts: int = 0
    total_rejects: int = 0
    total_end_calls: int = 0
    total_dnc_violations: int = 0
    total_tool_calls: int = 0
    total_tool_errors: int = 0
    total_random_events: int = 0

    # Sums for averages
    total_call_duration: int = 0
    total_revenue: float = 0.0
    total_api_cost: float = 0.0

    # Lists for distributions
    call_durations: list[int] = field(default_factory=list)
    offers_per_call: list[int] = field(default_factory=list)
    premiums_offered: list[float] = field(default_factory=list)
    premiums_accepted: list[float] = field(default_factory=list)

    # By-day tracking
    calls_by_day: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    accepts_by_day: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Current call state
    _current_call_offers: int = 0

    def on_event(self, event: Event) -> None:
        """Process an event."""
        if event.event_type == EventType.CALL_START:
            self.total_calls += 1
            self._current_call_offers = 0

        elif event.event_type == EventType.CALL_END:
            duration = event.data.get("duration_minutes", 0)
            self.total_call_duration += duration
            self.call_durations.append(duration)
            self.offers_per_call.append(self._current_call_offers)

        elif event.event_type == EventType.OFFER_PRESENTED:
            self.total_offers += 1
            self._current_call_offers += 1
            premium = event.data.get("offer", {}).get("monthly_premium", 0)
            self.premiums_offered.append(premium)

        elif event.event_type == EventType.BUYER_DECISION:
            decision = event.data.get("decision", "")
            if decision == "accept_plan":
                self.total_accepts += 1
                premium = event.data.get("offer", {}).get("monthly_premium", 0)
                self.premiums_accepted.append(premium)
                self.total_revenue += premium * 12  # Annualized
            elif decision == "reject_plan":
                self.total_rejects += 1
            elif decision == "end_call":
                self.total_end_calls += 1

        elif event.event_type == EventType.TOOL_CALL:
            self.total_tool_calls += 1

        elif event.event_type == EventType.TOOL_ERROR:
            self.total_tool_errors += 1

        elif event.event_type == EventType.RANDOM_EVENT:
            self.total_random_events += 1

        elif event.event_type == EventType.DAY_START:
            day = event.data.get("day", 0)
            # Initialize day tracking

    def aggregate(self) -> dict[str, Any]:
        """Compute episode metrics."""
        metrics = {
            # Counts
            "total_calls": self.total_calls,
            "total_offers": self.total_offers,
            "total_accepts": self.total_accepts,
            "total_rejects": self.total_rejects,
            "total_end_calls": self.total_end_calls,
            "total_dnc_violations": self.total_dnc_violations,
            "total_tool_calls": self.total_tool_calls,
            "total_tool_errors": self.total_tool_errors,
            "total_random_events": self.total_random_events,
            # Rates
            "acceptance_rate": self._safe_div(self.total_accepts, self.total_offers),
            "rejection_rate": self._safe_div(self.total_rejects, self.total_offers),
            "call_end_rate": self._safe_div(self.total_end_calls, self.total_calls),
            "tool_error_rate": self._safe_div(self.total_tool_errors, self.total_tool_calls),
            # Averages
            "avg_call_duration": self._safe_div(self.total_call_duration, self.total_calls),
            "avg_offers_per_call": self._safe_div(self.total_offers, self.total_calls),
            # Revenue
            "total_revenue": self.total_revenue,
            "revenue_per_call": self._safe_div(self.total_revenue, self.total_calls),
            "revenue_per_accept": self._safe_div(self.total_revenue, max(1, self.total_accepts)),
            # Premium stats
            "avg_premium_offered": self._list_avg(self.premiums_offered),
            "avg_premium_accepted": self._list_avg(self.premiums_accepted),
        }

        return metrics

    def reset(self) -> None:
        """Reset all counters."""
        self.total_calls = 0
        self.total_offers = 0
        self.total_accepts = 0
        self.total_rejects = 0
        self.total_end_calls = 0
        self.total_dnc_violations = 0
        self.total_tool_calls = 0
        self.total_tool_errors = 0
        self.total_random_events = 0
        self.total_call_duration = 0
        self.total_revenue = 0.0
        self.total_api_cost = 0.0
        self.call_durations.clear()
        self.offers_per_call.clear()
        self.premiums_offered.clear()
        self.premiums_accepted.clear()
        self.calls_by_day.clear()
        self.accepts_by_day.clear()
        self._current_call_offers = 0

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        """Safe division."""
        return a / b if b > 0 else 0.0

    @staticmethod
    def _list_avg(lst: list[float]) -> float:
        """Average of list."""
        return sum(lst) / len(lst) if lst else 0.0


@dataclass
class CallAggregator(EventAggregator):
    """Aggregates metrics for a single call."""

    call_id: Optional[str] = None
    lead_id: Optional[str] = None

    start_turn: int = 0
    end_turn: int = 0
    duration_minutes: int = 0

    offers: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    random_events: list[dict] = field(default_factory=list)

    outcome: Optional[str] = None

    def on_event(self, event: Event) -> None:
        """Process an event."""
        if event.event_type == EventType.CALL_START:
            self.call_id = event.call_id
            self.lead_id = event.lead_id
            self.start_turn = event.turn or 0

        elif event.event_type == EventType.CALL_END:
            self.end_turn = event.turn or 0
            self.duration_minutes = event.data.get("duration_minutes", 0)
            self.outcome = event.data.get("outcome")

        elif event.event_type == EventType.OFFER_PRESENTED:
            self.offers.append(event.data.get("offer", {}))

        elif event.event_type == EventType.BUYER_DECISION:
            self.decisions.append(
                {
                    "decision": event.data.get("decision"),
                    "reason": event.data.get("reason"),
                    "offer": event.data.get("offer"),
                }
            )

        elif event.event_type == EventType.RANDOM_EVENT:
            self.random_events.append(
                {
                    "type": event.data.get("random_event_type"),
                    "impacts": event.data.get("impacts"),
                }
            )

    def aggregate(self) -> dict[str, Any]:
        """Compute call metrics."""
        final_decision = self.decisions[-1]["decision"] if self.decisions else None

        return {
            "call_id": self.call_id,
            "lead_id": self.lead_id,
            "duration_minutes": self.duration_minutes,
            "turns": self.end_turn - self.start_turn,
            "offers_count": len(self.offers),
            "random_events_count": len(self.random_events),
            "outcome": self.outcome,
            "final_decision": final_decision,
            "accepted": final_decision == "accept_plan",
            "total_premium_offered": sum(o.get("monthly_premium", 0) for o in self.offers),
        }

    def reset(self) -> None:
        """Reset state."""
        self.call_id = None
        self.lead_id = None
        self.start_turn = 0
        self.end_turn = 0
        self.duration_minutes = 0
        self.offers.clear()
        self.decisions.clear()
        self.random_events.clear()
        self.outcome = None


@dataclass
class PerformanceAggregator(EventAggregator):
    """Aggregates performance metrics for pass^k analysis."""

    # Per-episode results
    episode_results: list[dict] = field(default_factory=list)

    # Current episode
    _current_episode_id: Optional[str] = None
    _current_aggregator: Optional[EpisodeAggregator] = None

    def on_event(self, event: Event) -> None:
        """Process an event."""
        if event.event_type == EventType.EPISODE_START:
            self._current_episode_id = event.episode_id
            self._current_aggregator = EpisodeAggregator()

        elif event.event_type == EventType.EPISODE_END:
            if self._current_aggregator:
                result = self._current_aggregator.aggregate()
                result["episode_id"] = self._current_episode_id
                result["seed"] = event.data.get("seed")
                result["total_reward"] = event.data.get("total_reward", 0)
                self.episode_results.append(result)

            self._current_episode_id = None
            self._current_aggregator = None

        elif self._current_aggregator:
            self._current_aggregator.on_event(event)

    def aggregate(self) -> dict[str, Any]:
        """Compute aggregate performance metrics."""
        if not self.episode_results:
            return {"episodes": 0}

        n = len(self.episode_results)

        # Success metrics
        accepts = [r["total_accepts"] for r in self.episode_results]
        revenues = [r["total_revenue"] for r in self.episode_results]
        acceptance_rates = [r["acceptance_rate"] for r in self.episode_results]

        return {
            "episodes": n,
            # Success rates
            "mean_accepts": sum(accepts) / n,
            "max_accepts": max(accepts),
            "min_accepts": min(accepts),
            # Revenue
            "mean_revenue": sum(revenues) / n,
            "max_revenue": max(revenues),
            "min_revenue": min(revenues),
            # Acceptance rates
            "mean_acceptance_rate": sum(acceptance_rates) / n,
            # pass^k metrics
            "pass_at_1": self._pass_at_k(1),
            "pass_at_3": self._pass_at_k(3),
            "pass_at_5": self._pass_at_k(5),
            "pass_at_10": self._pass_at_k(10),
            # Distribution of results
            "episode_results": self.episode_results,
        }

    def _pass_at_k(self, k: int, threshold: float = 0.0) -> float:
        """Compute pass@k metric.

        pass@k = probability that at least one of k samples achieves threshold.

        Args:
            k: Number of samples.
            threshold: Minimum acceptance rate to consider "passing".

        Returns:
            pass@k probability.
        """
        if not self.episode_results:
            return 0.0

        n = len(self.episode_results)
        if k > n:
            k = n

        # Count passing samples
        passing = sum(1 for r in self.episode_results if r.get("acceptance_rate", 0) > threshold)

        # Compute pass@k using combinatorial formula
        if passing == 0:
            return 0.0
        if passing >= k:
            return 1.0

        # P(at least one success in k) = 1 - P(no success in k)
        # = 1 - C(n-passing, k) / C(n, k)
        from math import comb

        prob_no_success = comb(n - passing, k) / comb(n, k) if n >= k else 0
        return 1.0 - prob_no_success

    def reset(self) -> None:
        """Reset all state."""
        self.episode_results.clear()
        self._current_episode_id = None
        self._current_aggregator = None

    def compute_pass_at_k(
        self,
        k_values: list[int],
        threshold: float = 0.0,
    ) -> dict[str, float]:
        """Compute pass@k for multiple k values.

        Args:
            k_values: List of k values.
            threshold: Minimum rate to consider passing.

        Returns:
            Dict mapping k to pass@k values.
        """
        return {f"pass@{k}": self._pass_at_k(k, threshold) for k in k_values}
