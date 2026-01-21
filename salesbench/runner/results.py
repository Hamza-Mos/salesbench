"""Result types for benchmark runs.

Defines EpisodeResult and BenchmarkResult for tracking and reporting
benchmark execution results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TokenUsage:
    """Tracks token usage for cost estimation.

    Attributes:
        seller_input_tokens: Input tokens used by seller LLM.
        seller_output_tokens: Output tokens used by seller LLM.
        buyer_input_tokens: Input tokens used by buyer LLM.
        buyer_output_tokens: Output tokens used by buyer LLM.
    """

    seller_input_tokens: int = 0
    seller_output_tokens: int = 0
    buyer_input_tokens: int = 0
    buyer_output_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all LLMs."""
        return self.seller_input_tokens + self.buyer_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all LLMs."""
        return self.seller_output_tokens + self.buyer_output_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def add_seller_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Add seller LLM usage."""
        self.seller_input_tokens += input_tokens
        self.seller_output_tokens += output_tokens

    def add_buyer_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Add buyer LLM usage."""
        self.buyer_input_tokens += input_tokens
        self.buyer_output_tokens += output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seller_input_tokens": self.seller_input_tokens,
            "seller_output_tokens": self.seller_output_tokens,
            "buyer_input_tokens": self.buyer_input_tokens,
            "buyer_output_tokens": self.buyer_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances."""
        return TokenUsage(
            seller_input_tokens=self.seller_input_tokens + other.seller_input_tokens,
            seller_output_tokens=self.seller_output_tokens + other.seller_output_tokens,
            buyer_input_tokens=self.buyer_input_tokens + other.buyer_input_tokens,
            buyer_output_tokens=self.buyer_output_tokens + other.buyer_output_tokens,
        )


@dataclass
class CostBreakdown:
    """Tracks cost breakdown by role and token type.

    Attributes:
        seller_input_cost: Cost of seller input tokens in USD.
        seller_output_cost: Cost of seller output tokens in USD.
        buyer_input_cost: Cost of buyer input tokens in USD.
        buyer_output_cost: Cost of buyer output tokens in USD.
        seller_pricing_available: Whether pricing info was available for seller model.
        buyer_pricing_available: Whether pricing info was available for buyer model.
    """

    seller_input_cost: float = 0.0
    seller_output_cost: float = 0.0
    buyer_input_cost: float = 0.0
    buyer_output_cost: float = 0.0
    seller_pricing_available: bool = False
    buyer_pricing_available: bool = False

    @property
    def seller_total_cost(self) -> float:
        """Total cost for seller."""
        return self.seller_input_cost + self.seller_output_cost

    @property
    def buyer_total_cost(self) -> float:
        """Total cost for buyer."""
        return self.buyer_input_cost + self.buyer_output_cost

    @property
    def total_input_cost(self) -> float:
        """Total input token cost."""
        return self.seller_input_cost + self.buyer_input_cost

    @property
    def total_output_cost(self) -> float:
        """Total output token cost."""
        return self.seller_output_cost + self.buyer_output_cost

    @property
    def total_cost(self) -> float:
        """Total cost (all tokens)."""
        return self.total_input_cost + self.total_output_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seller_input_cost": self.seller_input_cost,
            "seller_output_cost": self.seller_output_cost,
            "buyer_input_cost": self.buyer_input_cost,
            "buyer_output_cost": self.buyer_output_cost,
            "seller_total_cost": self.seller_total_cost,
            "buyer_total_cost": self.buyer_total_cost,
            "total_input_cost": self.total_input_cost,
            "total_output_cost": self.total_output_cost,
            "total_cost": self.total_cost,
            "seller_pricing_available": self.seller_pricing_available,
            "buyer_pricing_available": self.buyer_pricing_available,
        }

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        """Add two CostBreakdown instances."""
        return CostBreakdown(
            seller_input_cost=self.seller_input_cost + other.seller_input_cost,
            seller_output_cost=self.seller_output_cost + other.seller_output_cost,
            buyer_input_cost=self.buyer_input_cost + other.buyer_input_cost,
            buyer_output_cost=self.buyer_output_cost + other.buyer_output_cost,
            # AND logic: all episodes must have pricing for aggregate to be complete
            seller_pricing_available=self.seller_pricing_available
            and other.seller_pricing_available,
            buyer_pricing_available=self.buyer_pricing_available and other.buyer_pricing_available,
        )


def calculate_cost_breakdown(
    token_usage: TokenUsage,
    seller_model: str,
    buyer_model: str,
) -> CostBreakdown:
    """Calculate cost breakdown from token usage and model names.

    Args:
        token_usage: Token counts for seller and buyer.
        seller_model: Model name for seller (e.g., "gpt-4o").
        buyer_model: Model name for buyer (e.g., "gpt-4o-mini").

    Returns:
        CostBreakdown with costs calculated from model pricing.
    """
    from salesbench.models import SUPPORTED_MODELS

    breakdown = CostBreakdown()

    # Calculate seller costs
    seller_config = SUPPORTED_MODELS.get(seller_model)
    if seller_config and seller_config.input_price_per_million is not None:
        breakdown.seller_input_cost = (
            token_usage.seller_input_tokens * seller_config.input_price_per_million / 1_000_000
        )
        breakdown.seller_output_cost = (
            token_usage.seller_output_tokens * seller_config.output_price_per_million / 1_000_000
        )
        breakdown.seller_pricing_available = True

    # Calculate buyer costs
    buyer_config = SUPPORTED_MODELS.get(buyer_model)
    if buyer_config and buyer_config.input_price_per_million is not None:
        breakdown.buyer_input_cost = (
            token_usage.buyer_input_tokens * buyer_config.input_price_per_million / 1_000_000
        )
        breakdown.buyer_output_cost = (
            token_usage.buyer_output_tokens * buyer_config.output_price_per_million / 1_000_000
        )
        breakdown.buyer_pricing_available = True

    return breakdown


@dataclass
class EpisodeResult:
    """Result of a single episode execution.

    Attributes:
        episode_id: Unique identifier for this episode.
        benchmark_id: Parent benchmark ID.
        episode_index: Zero-based index within benchmark.
        seed: Random seed used for this episode.
        status: Execution status (completed, failed, timeout).
        final_score: Final score from the episode.
        total_turns: Number of turns executed.
        total_accepts: Number of accepted offers.
        total_rejects: Number of rejected offers.
        total_calls: Number of calls made.
        dnc_violations: Number of DNC violations.
        termination_reason: Why the episode ended.
        duration_seconds: Wall-clock duration.
        started_at: When episode started.
        ended_at: When episode ended.
        error: Error message if failed.
        metrics: Full metrics dict from orchestrator.
        token_usage: Token usage for cost estimation.
        cost_breakdown: Cost breakdown by role and token type.
    """

    episode_id: str
    benchmark_id: str
    episode_index: int
    seed: int
    status: str = "completed"
    final_score: float = 0.0
    total_turns: int = 0
    total_accepts: int = 0
    total_rejects: int = 0
    total_calls: int = 0
    dnc_violations: int = 0
    termination_reason: str = ""
    duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    trajectory: list[dict] = field(default_factory=list)  # Full conversation history
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_breakdown: CostBreakdown = field(default_factory=CostBreakdown)

    @property
    def succeeded(self) -> bool:
        """Check if episode completed successfully."""
        return self.status == "completed" and self.error is None

    @property
    def has_accepts(self) -> bool:
        """Check if episode had any acceptances."""
        return self.total_accepts > 0

    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate."""
        total_decisions = self.total_accepts + self.total_rejects
        if total_decisions == 0:
            return 0.0
        return self.total_accepts / total_decisions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "benchmark_id": self.benchmark_id,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "status": self.status,
            "final_score": self.final_score,
            "total_turns": self.total_turns,
            "total_accepts": self.total_accepts,
            "total_rejects": self.total_rejects,
            "total_calls": self.total_calls,
            "dnc_violations": self.dnc_violations,
            "termination_reason": self.termination_reason,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "error": self.error,
            "acceptance_rate": self.acceptance_rate,
            "metrics": self.metrics,
            "trajectory": self.trajectory,
            "token_usage": self.token_usage.to_dict(),
            "cost_breakdown": self.cost_breakdown.to_dict(),
        }


@dataclass
class BenchmarkResult:
    """Aggregate result of a complete benchmark run.

    Attributes:
        benchmark_id: Unique benchmark identifier.
        name: Benchmark name.
        mode: Run mode used.
        config: Benchmark configuration.
        total_episodes: Total episodes attempted.
        completed_episodes: Episodes that completed successfully.
        failed_episodes: Episodes that failed.
        episode_results: Individual episode results.
        started_at: When benchmark started.
        ended_at: When benchmark ended.
        duration_seconds: Total wall-clock duration.
        aggregate_metrics: Computed aggregate statistics.
    """

    benchmark_id: str
    name: str
    mode: str
    config: dict
    total_episodes: int = 0
    completed_episodes: int = 0
    failed_episodes: int = 0
    episode_results: list[EpisodeResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    aggregate_metrics: dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Percentage of episodes that completed successfully."""
        if self.total_episodes == 0:
            return 0.0
        return self.completed_episodes / self.total_episodes

    @property
    def mean_score(self) -> float:
        """Mean score across completed episodes."""
        completed = [e for e in self.episode_results if e.succeeded]
        if not completed:
            return 0.0
        return sum(e.final_score for e in completed) / len(completed)

    @property
    def std_score(self) -> float:
        """Standard deviation of scores."""
        completed = [e for e in self.episode_results if e.succeeded]
        if len(completed) < 2:
            return 0.0
        import statistics

        return statistics.stdev(e.final_score for e in completed)

    @property
    def total_accepts(self) -> int:
        """Total accepts across all episodes."""
        return sum(e.total_accepts for e in self.episode_results if e.succeeded)

    @property
    def mean_acceptance_rate(self) -> float:
        """Mean acceptance rate across completed episodes."""
        completed = [e for e in self.episode_results if e.succeeded]
        if not completed:
            return 0.0
        return sum(e.acceptance_rate for e in completed) / len(completed)

    @property
    def episodes_with_accepts(self) -> int:
        """Number of episodes that had at least one acceptance."""
        return sum(1 for e in self.episode_results if e.succeeded and e.has_accepts)

    def add_episode_result(self, result: EpisodeResult) -> None:
        """Add an episode result to the benchmark.

        Args:
            result: The episode result to add.
        """
        self.episode_results.append(result)
        self.total_episodes += 1
        if result.succeeded:
            self.completed_episodes += 1
        else:
            self.failed_episodes += 1

    def compute_aggregate_metrics(self) -> dict[str, Any]:
        """Compute aggregate metrics from episode results.

        Returns:
            Dictionary of aggregate statistics.
        """
        completed = [e for e in self.episode_results if e.succeeded]
        n = len(completed)

        if n == 0:
            return {
                "n_episodes": 0,
                "mean_score": 0.0,
                "std_score": 0.0,
            }

        import statistics

        from salesbench.metrics.pass_at_k import compute_pass_at_k

        scores = [e.final_score for e in completed]
        acceptance_rates = [e.acceptance_rate for e in completed]
        durations = [e.duration_seconds for e in completed]

        # Count episodes with at least one accept
        c = sum(1 for e in completed if e.has_accepts)

        # Aggregate token usage across all episodes
        total_tokens = TokenUsage()
        for e in self.episode_results:
            total_tokens = total_tokens + e.token_usage

        # Aggregate cost breakdown across all episodes
        total_cost = CostBreakdown(
            seller_pricing_available=True,
            buyer_pricing_available=True,
        )
        for e in self.episode_results:
            total_cost = total_cost + e.cost_breakdown

        metrics = {
            "n_episodes": n,
            "completed_episodes": self.completed_episodes,
            "failed_episodes": self.failed_episodes,
            # Score statistics
            "mean_score": statistics.mean(scores),
            "std_score": statistics.stdev(scores) if n > 1 else 0.0,
            "min_score": min(scores),
            "max_score": max(scores),
            "median_score": statistics.median(scores),
            # Acceptance statistics
            "total_accepts": self.total_accepts,
            "mean_acceptance_rate": statistics.mean(acceptance_rates),
            "std_acceptance_rate": statistics.stdev(acceptance_rates) if n > 1 else 0.0,
            "episodes_with_accepts": c,
            "episode_success_rate": c / n if n > 0 else 0.0,
            # Pass@k metrics
            "pass_at_1": compute_pass_at_k(n, c, 1),
            "pass_at_5": compute_pass_at_k(n, c, 5) if n >= 5 else None,
            "pass_at_10": compute_pass_at_k(n, c, 10) if n >= 10 else None,
            "pass_at_100": compute_pass_at_k(n, c, 100) if n >= 100 else None,
            # Timing
            "total_duration_seconds": sum(durations),
            "mean_episode_duration": statistics.mean(durations),
            # Token usage
            "total_token_usage": total_tokens.to_dict(),
            # Cost breakdown
            "total_cost_breakdown": total_cost.to_dict(),
        }

        self.aggregate_metrics = metrics
        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "mode": self.mode,
            "config": self.config,
            "total_episodes": self.total_episodes,
            "completed_episodes": self.completed_episodes,
            "failed_episodes": self.failed_episodes,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "aggregate_metrics": self.aggregate_metrics,
            "episode_results": [e.to_dict() for e in self.episode_results],
        }

    def to_summary_dict(self) -> dict[str, Any]:
        """Get a summary without individual episode details."""
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "mode": self.mode,
            "total_episodes": self.total_episodes,
            "completed_episodes": self.completed_episodes,
            "failed_episodes": self.failed_episodes,
            "duration_seconds": self.duration_seconds,
            "aggregate_metrics": self.aggregate_metrics,
        }
