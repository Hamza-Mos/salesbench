"""Metric hooks for integration with evaluation frameworks.

Hooks provide callbacks that are invoked at various points during
benchmark execution, allowing metrics to be collected and reported
to external systems like Prime Intellect.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class MetricHook(ABC):
    """Abstract base class for metric hooks."""

    @abstractmethod
    def on_episode_start(
        self,
        episode_id: str,
        seed: int,
        config: dict[str, Any],
    ) -> None:
        """Called when an episode starts.

        Args:
            episode_id: Unique episode identifier.
            seed: Random seed for the episode.
            config: Episode configuration.
        """
        pass

    @abstractmethod
    def on_episode_end(
        self,
        episode_id: str,
        result: dict[str, Any],
    ) -> None:
        """Called when an episode ends.

        Args:
            episode_id: Unique episode identifier.
            result: Episode result metrics.
        """
        pass

    def on_turn(
        self,
        episode_id: str,
        turn: int,
        action: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        """Called after each turn (optional).

        Args:
            episode_id: Unique episode identifier.
            turn: Turn number.
            action: Agent action.
            observation: Environment observation.
        """
        pass

    def on_benchmark_start(
        self,
        benchmark_id: str,
        config: dict[str, Any],
    ) -> None:
        """Called when a benchmark run starts.

        Args:
            benchmark_id: Unique benchmark identifier.
            config: Benchmark configuration.
        """
        pass

    def on_benchmark_end(
        self,
        benchmark_id: str,
        results: dict[str, Any],
    ) -> None:
        """Called when a benchmark run ends.

        Args:
            benchmark_id: Unique benchmark identifier.
            results: Aggregate benchmark results.
        """
        pass


@dataclass
class EpisodeMetricHook(MetricHook):
    """Hook that collects metrics for each episode."""

    episode_results: list[dict] = field(default_factory=list)
    current_episode: Optional[dict] = None
    verbose: bool = False

    def on_episode_start(
        self,
        episode_id: str,
        seed: int,
        config: dict[str, Any],
    ) -> None:
        self.current_episode = {
            "episode_id": episode_id,
            "seed": seed,
            "config": config,
            "start_time": time.time(),
            "turns": 0,
        }
        if self.verbose:
            print(f"Episode {episode_id} started (seed={seed})")

    def on_episode_end(
        self,
        episode_id: str,
        result: dict[str, Any],
    ) -> None:
        if self.current_episode and self.current_episode["episode_id"] == episode_id:
            self.current_episode["end_time"] = time.time()
            self.current_episode["duration"] = (
                self.current_episode["end_time"] - self.current_episode["start_time"]
            )
            self.current_episode["result"] = result
            self.episode_results.append(self.current_episode)

            if self.verbose:
                print(
                    f"Episode {episode_id} ended: "
                    f"accepts={result.get('total_accepts', 0)}, "
                    f"revenue=${result.get('total_revenue', 0):.2f}"
                )

        self.current_episode = None

    def on_turn(
        self,
        episode_id: str,
        turn: int,
        action: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        if self.current_episode:
            self.current_episode["turns"] = turn

    def get_results(self) -> list[dict]:
        """Get all episode results."""
        return self.episode_results

    def reset(self) -> None:
        """Reset collected results."""
        self.episode_results.clear()
        self.current_episode = None


class PassAtKHook(MetricHook):
    """Hook that computes pass@k metrics as episodes complete."""

    def __init__(
        self,
        k_values: list[int] = None,
        success_threshold: float = 0.0,
        report_interval: int = 10,
        on_report: Optional[Callable[[dict], None]] = None,
    ):
        """Initialize the hook.

        Args:
            k_values: k values to compute.
            success_threshold: Minimum acceptance rate for success.
            report_interval: Report every N episodes.
            on_report: Optional callback for reports.
        """
        self.k_values = k_values or [1, 5, 10, 50, 100]
        self.success_threshold = success_threshold
        self.report_interval = report_interval
        self.on_report = on_report

        self._results: list[dict] = []
        self._current_episode: Optional[str] = None

    def on_episode_start(
        self,
        episode_id: str,
        seed: int,
        config: dict[str, Any],
    ) -> None:
        self._current_episode = episode_id

    def on_episode_end(
        self,
        episode_id: str,
        result: dict[str, Any],
    ) -> None:
        self._results.append(result)
        self._current_episode = None

        # Report periodically
        if len(self._results) % self.report_interval == 0:
            self._report()

    def on_benchmark_end(
        self,
        benchmark_id: str,
        results: dict[str, Any],
    ) -> None:
        # Final report
        self._report()

    def _report(self) -> None:
        """Generate and optionally send a report."""
        if not self._results:
            return

        n = len(self._results)
        c = sum(1 for r in self._results if r.get("acceptance_rate", 0) > self.success_threshold)

        # Import here to avoid circular dependency
        from salesbench.metrics.pass_at_k import compute_pass_at_k

        report = {
            "n": n,
            "c": c,
            "pass_rate": c / n if n > 0 else 0,
            "pass_at_k": {k: compute_pass_at_k(n, c, k) for k in self.k_values},
            "mean_acceptance_rate": (sum(r.get("acceptance_rate", 0) for r in self._results) / n),
            "mean_revenue": (sum(r.get("total_revenue", 0) for r in self._results) / n),
        }

        if self.on_report:
            self.on_report(report)

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current pass@k metrics."""
        if not self._results:
            return {"n": 0, "pass_at_k": {k: 0.0 for k in self.k_values}}

        from salesbench.metrics.pass_at_k import compute_pass_at_k

        n = len(self._results)
        c = sum(1 for r in self._results if r.get("acceptance_rate", 0) > self.success_threshold)

        return {
            "n": n,
            "c": c,
            "pass_rate": c / n,
            "pass_at_k": {k: compute_pass_at_k(n, c, k) for k in self.k_values},
        }

    def reset(self) -> None:
        """Reset collected results."""
        self._results.clear()
        self._current_episode = None


class PrimeIntellectHook(MetricHook):
    """Hook for Prime Intellect Verifiers integration.

    Formats and reports metrics in the format expected by
    the Prime Intellect evaluation framework.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 10,
    ):
        """Initialize the hook.

        Args:
            endpoint_url: URL to POST metrics to (optional).
            api_key: API key for authentication.
            batch_size: Number of results to batch before sending.
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.batch_size = batch_size

        self._batch: list[dict] = []
        self._benchmark_id: Optional[str] = None

    def on_benchmark_start(
        self,
        benchmark_id: str,
        config: dict[str, Any],
    ) -> None:
        self._benchmark_id = benchmark_id
        self._batch.clear()

    def on_episode_start(
        self,
        episode_id: str,
        seed: int,
        config: dict[str, Any],
    ) -> None:
        pass  # No action needed

    def on_episode_end(
        self,
        episode_id: str,
        result: dict[str, Any],
    ) -> None:
        # Format result for Prime Intellect
        formatted = self._format_result(episode_id, result)
        self._batch.append(formatted)

        # Send batch if full
        if len(self._batch) >= self.batch_size:
            self._send_batch()

    def on_benchmark_end(
        self,
        benchmark_id: str,
        results: dict[str, Any],
    ) -> None:
        # Send remaining batch
        if self._batch:
            self._send_batch()

        # Send final summary
        self._send_summary(benchmark_id, results)

    def _format_result(
        self,
        episode_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Format a result for Prime Intellect."""
        return {
            "benchmark_id": self._benchmark_id,
            "episode_id": episode_id,
            "timestamp": time.time(),
            "metrics": {
                "acceptance_rate": result.get("acceptance_rate", 0),
                "total_revenue": result.get("total_revenue", 0),
                "total_accepts": result.get("total_accepts", 0),
                "total_rejects": result.get("total_rejects", 0),
                "total_calls": result.get("total_calls", 0),
                "avg_call_duration": result.get("avg_call_duration", 0),
            },
            "success": result.get("acceptance_rate", 0) > 0,
        }

    def _send_batch(self) -> None:
        """Send a batch of results."""
        if not self.endpoint_url:
            # Just log locally
            for result in self._batch:
                print(f"[PI] Episode {result['episode_id']}: {result['metrics']}")
            self._batch.clear()
            return

        try:
            import requests

            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(
                f"{self.endpoint_url}/results",
                headers=headers,
                json={"results": self._batch},
                timeout=30,
            )
            response.raise_for_status()

        except Exception as e:
            print(f"[PI] Warning: Failed to send batch: {e}")

        self._batch.clear()

    def _send_summary(
        self,
        benchmark_id: str,
        results: dict[str, Any],
    ) -> None:
        """Send benchmark summary."""
        summary = {
            "benchmark_id": benchmark_id,
            "timestamp": time.time(),
            "summary": results,
        }

        if not self.endpoint_url:
            print(f"[PI] Benchmark {benchmark_id} complete: {json.dumps(results, indent=2)}")
            return

        try:
            import requests

            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(
                f"{self.endpoint_url}/summary",
                headers=headers,
                json=summary,
                timeout=30,
            )
            response.raise_for_status()

        except Exception as e:
            print(f"[PI] Warning: Failed to send summary: {e}")


class CompositeHook(MetricHook):
    """Combines multiple hooks."""

    def __init__(self, hooks: list[MetricHook]):
        """Initialize with multiple hooks.

        Args:
            hooks: List of hooks to compose.
        """
        self.hooks = hooks

    def on_benchmark_start(
        self,
        benchmark_id: str,
        config: dict[str, Any],
    ) -> None:
        for hook in self.hooks:
            hook.on_benchmark_start(benchmark_id, config)

    def on_episode_start(
        self,
        episode_id: str,
        seed: int,
        config: dict[str, Any],
    ) -> None:
        for hook in self.hooks:
            hook.on_episode_start(episode_id, seed, config)

    def on_turn(
        self,
        episode_id: str,
        turn: int,
        action: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        for hook in self.hooks:
            hook.on_turn(episode_id, turn, action, observation)

    def on_episode_end(
        self,
        episode_id: str,
        result: dict[str, Any],
    ) -> None:
        for hook in self.hooks:
            hook.on_episode_end(episode_id, result)

    def on_benchmark_end(
        self,
        benchmark_id: str,
        results: dict[str, Any],
    ) -> None:
        for hook in self.hooks:
            hook.on_benchmark_end(benchmark_id, results)
