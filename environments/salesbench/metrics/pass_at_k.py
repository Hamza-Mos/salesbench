"""pass@k computation for SalesBench evaluation.

pass@k measures the probability that at least one of k independent samples
achieves a success criterion. This is the standard evaluation metric for
code generation benchmarks and is adapted here for sales agent evaluation.

For SalesBench:
- Success = achieving a minimum acceptance rate or revenue target
- k = number of independent episode runs
- Used for comparing different agent architectures

Mathematical definition:
pass@k = 1 - (n-c choose k) / (n choose k)

Where:
- n = total samples
- c = number of correct (passing) samples
- k = samples to consider

For large n, this can be estimated using:
pass@k ≈ 1 - (1 - c/n)^k
"""

from dataclasses import dataclass, field
from math import comb
from typing import Any, Callable, Optional
import numpy as np


@dataclass
class PassAtKConfig:
    """Configuration for pass@k computation."""

    # Success criteria
    min_acceptance_rate: float = 0.0  # Minimum rate to count as passing
    min_revenue: float = 0.0  # Minimum revenue to count as passing
    min_accepts: int = 0  # Minimum accepts to count as passing

    # Which criteria to use (any/all)
    require_all_criteria: bool = False  # If True, all criteria must pass

    # k values to compute
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 25, 50, 100])

    # Estimation settings
    use_estimation: bool = False  # Use estimation for large n
    estimation_threshold: int = 1000  # Use estimation when n > threshold


@dataclass
class PassAtKResult:
    """Result of pass@k computation."""

    # Raw data
    n: int  # Total samples
    c: int  # Passing samples
    pass_rate: float  # c/n

    # pass@k values
    pass_at_k: dict[int, float]  # k -> pass@k value

    # Summary statistics
    mean_acceptance_rate: float
    std_acceptance_rate: float
    mean_revenue: float
    std_revenue: float

    # Best/worst results
    best_acceptance_rate: float
    worst_acceptance_rate: float
    best_revenue: float
    worst_revenue: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n": self.n,
            "c": self.c,
            "pass_rate": self.pass_rate,
            "pass_at_k": self.pass_at_k,
            "mean_acceptance_rate": self.mean_acceptance_rate,
            "std_acceptance_rate": self.std_acceptance_rate,
            "mean_revenue": self.mean_revenue,
            "std_revenue": self.std_revenue,
            "best_acceptance_rate": self.best_acceptance_rate,
            "worst_acceptance_rate": self.worst_acceptance_rate,
            "best_revenue": self.best_revenue,
            "worst_revenue": self.worst_revenue,
        }


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute exact pass@k using combinatorics.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total number of samples.
        c: Number of passing samples.
        k: Number of samples to consider.

    Returns:
        pass@k probability.
    """
    if n <= 0 or k <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if c == 0:
        return 0.0
    if k > n:
        k = n

    # Compute using logarithms for numerical stability
    # log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
    # But we use comb() for small values
    if n < 10000:
        try:
            num = comb(n - c, k)
            denom = comb(n, k)
            if denom == 0:
                return 0.0
            return 1.0 - (num / denom)
        except (ValueError, OverflowError):
            return estimate_pass_at_k(n, c, k)
    else:
        return estimate_pass_at_k(n, c, k)


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate pass@k for large n using probability theory.

    For large n:
    pass@k ≈ 1 - (1 - c/n)^k

    This is a good approximation when k << n.

    Args:
        n: Total number of samples.
        c: Number of passing samples.
        k: Number of samples to consider.

    Returns:
        Estimated pass@k probability.
    """
    if n <= 0 or k <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if c == 0:
        return 0.0

    p = c / n  # Probability of success
    return 1.0 - (1.0 - p) ** k


class PassAtKComputer:
    """Computes pass@k metrics for episode results."""

    def __init__(self, config: Optional[PassAtKConfig] = None):
        """Initialize the computer.

        Args:
            config: Configuration for pass@k computation.
        """
        self.config = config or PassAtKConfig()
        self._results: list[dict] = []

    def add_result(self, episode_result: dict) -> None:
        """Add an episode result.

        Args:
            episode_result: Dictionary with keys like:
                - acceptance_rate: float
                - total_revenue: float
                - total_accepts: int
                - etc.
        """
        self._results.append(episode_result)

    def add_results(self, results: list[dict]) -> None:
        """Add multiple episode results."""
        self._results.extend(results)

    def is_passing(self, result: dict) -> bool:
        """Check if a result passes the success criteria.

        Args:
            result: Episode result dictionary.

        Returns:
            True if result passes criteria.
        """
        criteria_met = []

        # Check acceptance rate
        if self.config.min_acceptance_rate > 0:
            rate = result.get("acceptance_rate", 0)
            criteria_met.append(rate >= self.config.min_acceptance_rate)

        # Check revenue
        if self.config.min_revenue > 0:
            revenue = result.get("total_revenue", 0)
            criteria_met.append(revenue >= self.config.min_revenue)

        # Check accepts
        if self.config.min_accepts > 0:
            accepts = result.get("total_accepts", 0)
            criteria_met.append(accepts >= self.config.min_accepts)

        if not criteria_met:
            # No criteria specified - consider all passing
            return True

        if self.config.require_all_criteria:
            return all(criteria_met)
        else:
            return any(criteria_met)

    def compute(self) -> PassAtKResult:
        """Compute pass@k metrics.

        Returns:
            PassAtKResult with all computed metrics.
        """
        if not self._results:
            return PassAtKResult(
                n=0,
                c=0,
                pass_rate=0.0,
                pass_at_k={k: 0.0 for k in self.config.k_values},
                mean_acceptance_rate=0.0,
                std_acceptance_rate=0.0,
                mean_revenue=0.0,
                std_revenue=0.0,
                best_acceptance_rate=0.0,
                worst_acceptance_rate=0.0,
                best_revenue=0.0,
                worst_revenue=0.0,
            )

        n = len(self._results)

        # Count passing results
        c = sum(1 for r in self._results if self.is_passing(r))
        pass_rate = c / n

        # Compute pass@k for all k values
        pass_at_k = {}
        for k in self.config.k_values:
            if k > n:
                # Can't compute pass@k for k > n
                pass_at_k[k] = pass_rate  # Best estimate
            elif self.config.use_estimation and n > self.config.estimation_threshold:
                pass_at_k[k] = estimate_pass_at_k(n, c, k)
            else:
                pass_at_k[k] = compute_pass_at_k(n, c, k)

        # Compute statistics
        acceptance_rates = [r.get("acceptance_rate", 0) for r in self._results]
        revenues = [r.get("total_revenue", 0) for r in self._results]

        return PassAtKResult(
            n=n,
            c=c,
            pass_rate=pass_rate,
            pass_at_k=pass_at_k,
            mean_acceptance_rate=np.mean(acceptance_rates) if acceptance_rates else 0.0,
            std_acceptance_rate=np.std(acceptance_rates) if acceptance_rates else 0.0,
            mean_revenue=np.mean(revenues) if revenues else 0.0,
            std_revenue=np.std(revenues) if revenues else 0.0,
            best_acceptance_rate=max(acceptance_rates) if acceptance_rates else 0.0,
            worst_acceptance_rate=min(acceptance_rates) if acceptance_rates else 0.0,
            best_revenue=max(revenues) if revenues else 0.0,
            worst_revenue=min(revenues) if revenues else 0.0,
        )

    def compute_stratified(
        self,
        stratify_by: str,
    ) -> dict[str, PassAtKResult]:
        """Compute pass@k stratified by a field.

        Args:
            stratify_by: Field to stratify by (e.g., "agent_type", "seed_range").

        Returns:
            Dictionary mapping stratum values to PassAtKResult.
        """
        strata: dict[str, list[dict]] = {}

        for result in self._results:
            key = str(result.get(stratify_by, "unknown"))
            if key not in strata:
                strata[key] = []
            strata[key].append(result)

        results = {}
        for key, stratum_results in strata.items():
            computer = PassAtKComputer(self.config)
            computer.add_results(stratum_results)
            results[key] = computer.compute()

        return results

    def reset(self) -> None:
        """Reset all results."""
        self._results.clear()


def compute_pass_at_k_batch(
    results: list[dict],
    k_values: list[int] = None,
    success_fn: Optional[Callable[[dict], bool]] = None,
) -> dict[int, float]:
    """Convenience function to compute pass@k for a batch of results.

    Args:
        results: List of episode results.
        k_values: k values to compute (default: [1, 5, 10, 50, 100]).
        success_fn: Optional custom success function.

    Returns:
        Dictionary mapping k to pass@k values.
    """
    k_values = k_values or [1, 5, 10, 50, 100]

    if success_fn:
        c = sum(1 for r in results if success_fn(r))
    else:
        # Default: any non-zero acceptance rate
        c = sum(1 for r in results if r.get("acceptance_rate", 0) > 0)

    n = len(results)

    return {k: compute_pass_at_k(n, c, k) for k in k_values}
