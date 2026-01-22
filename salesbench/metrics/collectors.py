"""Metric collectors for SalesBench.

Collectors aggregate metrics during benchmark runs and provide
structured summaries for reporting and analysis.
"""

import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    # Identity
    episode_id: str
    seed: int
    agent_type: str

    # Timing
    start_time: float
    end_time: float
    duration_seconds: float

    # Call metrics
    total_calls: int = 0
    total_call_duration: int = 0
    avg_call_duration: float = 0.0

    # Offer metrics
    total_offers: int = 0
    total_accepts: int = 0
    total_rejects: int = 0
    total_end_calls: int = 0

    # Rates
    acceptance_rate: float = 0.0
    rejection_rate: float = 0.0

    # Revenue
    total_revenue: float = 0.0
    revenue_per_call: float = 0.0
    revenue_per_accept: float = 0.0

    # Tool usage
    total_tool_calls: int = 0
    total_tool_errors: int = 0
    tool_error_rate: float = 0.0

    # Events
    total_random_events: int = 0
    total_dnc_violations: int = 0

    # Costs
    api_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "agent_type": self.agent_type,
            "duration_seconds": self.duration_seconds,
            "total_calls": self.total_calls,
            "total_call_duration": self.total_call_duration,
            "avg_call_duration": self.avg_call_duration,
            "total_offers": self.total_offers,
            "total_accepts": self.total_accepts,
            "total_rejects": self.total_rejects,
            "total_end_calls": self.total_end_calls,
            "acceptance_rate": self.acceptance_rate,
            "rejection_rate": self.rejection_rate,
            "total_revenue": self.total_revenue,
            "revenue_per_call": self.revenue_per_call,
            "revenue_per_accept": self.revenue_per_accept,
            "total_tool_calls": self.total_tool_calls,
            "total_tool_errors": self.total_tool_errors,
            "tool_error_rate": self.tool_error_rate,
            "total_random_events": self.total_random_events,
            "total_dnc_violations": self.total_dnc_violations,
            "api_cost": self.api_cost,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple episodes."""

    # Count
    n_episodes: int = 0

    # Acceptance rates
    mean_acceptance_rate: float = 0.0
    std_acceptance_rate: float = 0.0
    min_acceptance_rate: float = 0.0
    max_acceptance_rate: float = 0.0
    median_acceptance_rate: float = 0.0

    # Revenue
    mean_revenue: float = 0.0
    std_revenue: float = 0.0
    min_revenue: float = 0.0
    max_revenue: float = 0.0
    total_revenue: float = 0.0

    # Calls
    mean_calls: float = 0.0
    mean_call_duration: float = 0.0
    mean_offers_per_call: float = 0.0

    # Success rates
    episodes_with_accepts: int = 0
    success_rate: float = 0.0  # % episodes with at least one accept

    # Timing
    total_duration_seconds: float = 0.0
    mean_episode_duration: float = 0.0

    # Costs
    total_api_cost: float = 0.0
    mean_api_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_episodes": self.n_episodes,
            "mean_acceptance_rate": self.mean_acceptance_rate,
            "std_acceptance_rate": self.std_acceptance_rate,
            "min_acceptance_rate": self.min_acceptance_rate,
            "max_acceptance_rate": self.max_acceptance_rate,
            "median_acceptance_rate": self.median_acceptance_rate,
            "mean_revenue": self.mean_revenue,
            "std_revenue": self.std_revenue,
            "min_revenue": self.min_revenue,
            "max_revenue": self.max_revenue,
            "total_revenue": self.total_revenue,
            "mean_calls": self.mean_calls,
            "mean_call_duration": self.mean_call_duration,
            "mean_offers_per_call": self.mean_offers_per_call,
            "episodes_with_accepts": self.episodes_with_accepts,
            "success_rate": self.success_rate,
            "total_duration_seconds": self.total_duration_seconds,
            "mean_episode_duration": self.mean_episode_duration,
            "total_api_cost": self.total_api_cost,
            "mean_api_cost": self.mean_api_cost,
        }


class MetricCollector:
    """Collects and aggregates metrics during benchmark runs.

    Thread-safe for parallel episode collection.
    """

    def __init__(self, agent_type: str = "unknown"):
        """Initialize the collector.

        Args:
            agent_type: Type of agent being evaluated.
        """
        self.agent_type = agent_type
        self._episodes: list[EpisodeMetrics] = []
        self._current_episode: Optional[dict] = None
        self._lock = threading.Lock()

    def start_episode(self, episode_id: str, seed: int) -> None:
        """Start collecting metrics for a new episode.

        Args:
            episode_id: Unique episode identifier.
            seed: Random seed for the episode.
        """
        self._current_episode = {
            "episode_id": episode_id,
            "seed": seed,
            "start_time": time.time(),
            # Counters
            "calls": 0,
            "call_duration": 0,
            "offers": 0,
            "accepts": 0,
            "rejects": 0,
            "end_calls": 0,
            "revenue": 0.0,
            "tool_calls": 0,
            "tool_errors": 0,
            "random_events": 0,
            "dnc_violations": 0,
            "api_cost": 0.0,
        }

    def record_call(self, duration_minutes: int) -> None:
        """Record a completed call."""
        if self._current_episode:
            self._current_episode["calls"] += 1
            self._current_episode["call_duration"] += duration_minutes

    def record_offer(self) -> None:
        """Record an offer presented."""
        if self._current_episode:
            self._current_episode["offers"] += 1

    def record_decision(self, decision: str, premium: float = 0.0) -> None:
        """Record a buyer decision.

        Args:
            decision: The decision (accept_plan, reject_plan, end_call).
            premium: Monthly premium if accepted.
        """
        if not self._current_episode:
            return

        if decision == "accept_plan":
            self._current_episode["accepts"] += 1
            self._current_episode["revenue"] += premium * 12  # Annualized
        elif decision == "reject_plan":
            self._current_episode["rejects"] += 1
        elif decision == "end_call":
            self._current_episode["end_calls"] += 1

    def record_tool_call(self, success: bool) -> None:
        """Record a tool call."""
        if self._current_episode:
            self._current_episode["tool_calls"] += 1
            if not success:
                self._current_episode["tool_errors"] += 1

    def record_random_event(self) -> None:
        """Record a random event."""
        if self._current_episode:
            self._current_episode["random_events"] += 1

    def record_dnc_violation(self) -> None:
        """Record a DNC violation."""
        if self._current_episode:
            self._current_episode["dnc_violations"] += 1

    def record_api_cost(self, cost: float) -> None:
        """Record API cost."""
        if self._current_episode:
            self._current_episode["api_cost"] += cost

    def end_episode(self) -> EpisodeMetrics:
        """End the current episode and return metrics.

        Returns:
            EpisodeMetrics for the completed episode.
        """
        if not self._current_episode:
            raise ValueError("No active episode")

        ep = self._current_episode
        end_time = time.time()
        duration = end_time - ep["start_time"]

        # Calculate rates
        total_offers = ep["offers"]
        acceptance_rate = ep["accepts"] / total_offers if total_offers > 0 else 0
        rejection_rate = ep["rejects"] / total_offers if total_offers > 0 else 0

        # Calculate averages
        avg_call_duration = ep["call_duration"] / ep["calls"] if ep["calls"] > 0 else 0
        revenue_per_call = ep["revenue"] / ep["calls"] if ep["calls"] > 0 else 0
        revenue_per_accept = ep["revenue"] / ep["accepts"] if ep["accepts"] > 0 else 0
        tool_error_rate = ep["tool_errors"] / ep["tool_calls"] if ep["tool_calls"] > 0 else 0

        metrics = EpisodeMetrics(
            episode_id=ep["episode_id"],
            seed=ep["seed"],
            agent_type=self.agent_type,
            start_time=ep["start_time"],
            end_time=end_time,
            duration_seconds=duration,
            total_calls=ep["calls"],
            total_call_duration=ep["call_duration"],
            avg_call_duration=avg_call_duration,
            total_offers=ep["offers"],
            total_accepts=ep["accepts"],
            total_rejects=ep["rejects"],
            total_end_calls=ep["end_calls"],
            acceptance_rate=acceptance_rate,
            rejection_rate=rejection_rate,
            total_revenue=ep["revenue"],
            revenue_per_call=revenue_per_call,
            revenue_per_accept=revenue_per_accept,
            total_tool_calls=ep["tool_calls"],
            total_tool_errors=ep["tool_errors"],
            tool_error_rate=tool_error_rate,
            total_random_events=ep["random_events"],
            total_dnc_violations=ep["dnc_violations"],
            api_cost=ep["api_cost"],
        )

        with self._lock:
            self._episodes.append(metrics)
        self._current_episode = None

        return metrics

    def add_result(self, metrics: EpisodeMetrics) -> None:
        """Add a pre-computed episode result (thread-safe).

        Use this method for parallel episode collection where metrics
        are computed by separate executors.

        Args:
            metrics: Pre-computed episode metrics.
        """
        with self._lock:
            self._episodes.append(metrics)

    def get_aggregate_metrics(self) -> AggregateMetrics:
        """Compute aggregate metrics across all episodes.

        Returns:
            AggregateMetrics summary.
        """
        if not self._episodes:
            return AggregateMetrics()

        n = len(self._episodes)

        # Extract values for statistics
        acceptance_rates = [e.acceptance_rate for e in self._episodes]
        revenues = [e.total_revenue for e in self._episodes]
        calls = [e.total_calls for e in self._episodes]
        durations = [e.duration_seconds for e in self._episodes]
        api_costs = [e.api_cost for e in self._episodes]

        # Compute statistics
        mean_acceptance = statistics.mean(acceptance_rates)
        std_acceptance = statistics.stdev(acceptance_rates) if n > 1 else 0

        mean_revenue = statistics.mean(revenues)
        std_revenue = statistics.stdev(revenues) if n > 1 else 0

        # Count successful episodes
        episodes_with_accepts = sum(1 for e in self._episodes if e.total_accepts > 0)

        # Pre-compute totals
        total_calls = sum(calls)
        total_offers = sum(e.total_offers for e in self._episodes)

        return AggregateMetrics(
            n_episodes=n,
            mean_acceptance_rate=mean_acceptance,
            std_acceptance_rate=std_acceptance,
            min_acceptance_rate=min(acceptance_rates),
            max_acceptance_rate=max(acceptance_rates),
            median_acceptance_rate=statistics.median(acceptance_rates),
            mean_revenue=mean_revenue,
            std_revenue=std_revenue,
            min_revenue=min(revenues),
            max_revenue=max(revenues),
            total_revenue=sum(revenues),
            mean_calls=statistics.mean(calls),
            mean_call_duration=statistics.mean([e.avg_call_duration for e in self._episodes]),
            mean_offers_per_call=total_offers / total_calls if total_calls > 0 else 0,
            episodes_with_accepts=episodes_with_accepts,
            success_rate=episodes_with_accepts / n,
            total_duration_seconds=sum(durations),
            mean_episode_duration=statistics.mean(durations),
            total_api_cost=sum(api_costs),
            mean_api_cost=statistics.mean(api_costs),
        )

    def get_episode_results(self) -> list[dict]:
        """Get all episode results as dictionaries.

        Returns:
            List of episode metrics as dicts.
        """
        return [e.to_dict() for e in self._episodes]

    def reset(self) -> None:
        """Reset all collected metrics (thread-safe)."""
        with self._lock:
            self._episodes.clear()
            self._current_episode = None
