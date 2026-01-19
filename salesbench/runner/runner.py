"""Main benchmark runner with parallel execution.

Orchestrates running multiple episodes in parallel with telemetry
and storage integrations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional

from salesbench.runner.config import BenchmarkConfig, RunMode
from salesbench.runner.executor import EpisodeExecutor
from salesbench.runner.integrations import IntegrationManager
from salesbench.runner.results import BenchmarkResult, EpisodeResult

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main runner for SalesBench benchmarks.

    Runs multiple episodes in parallel using asyncio, collecting
    results and computing aggregate metrics.

    Example:
        config = BenchmarkConfig.from_mode(RunMode.TEST)
        runner = BenchmarkRunner(config)
        result = runner.run()
        print(result.aggregate_metrics)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        output_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration.
            output_callback: Optional callback for progress output.
        """
        self.config = config
        self.output_callback = output_callback or (lambda x: print(x))
        self.integrations = IntegrationManager(config)

    def run(self) -> BenchmarkResult:
        """Run the complete benchmark.

        Returns:
            BenchmarkResult with all episode results and aggregates.
        """
        return asyncio.run(self._run_async())

    async def _run_async(self) -> BenchmarkResult:
        """Async implementation of benchmark run."""
        started_at = datetime.utcnow()
        start_time = time.time()

        # Initialize result
        result = BenchmarkResult(
            benchmark_id=self.config.benchmark_id,
            name=self.config.name,
            mode=self.config.mode.value,
            config=self.config.to_dict(),
            started_at=started_at,
        )

        # Print header
        self._print_header()

        # Start integrations
        self.integrations.start()

        try:
            # Create agents
            seller_agent, buyer_simulator, display_seller, display_buyer = self._create_agents()

            if seller_agent is None:
                self.output_callback("ERROR: Failed to create agents")
                return result

            self.output_callback(f"Seller model: {display_seller}")
            self.output_callback(f"Buyer model: {display_buyer}")
            self.output_callback(
                f"Supabase: {'enabled' if self.integrations.supabase_enabled else 'disabled'}"
            )
            self.output_callback(
                f"Telemetry: {'enabled' if self.integrations.telemetry_enabled else 'disabled'}"
            )
            self.output_callback("")
            self.output_callback("Running episodes...")

            # Run episodes with parallelism
            episode_results = await self._run_all_episodes(
                seller_agent=seller_agent,
                buyer_simulator=buyer_simulator,
            )

            # Collect results
            for ep_result in episode_results:
                result.add_episode_result(ep_result)

            # Compute aggregates
            result.compute_aggregate_metrics()

            # Write benchmark summary
            self.integrations.write_benchmark_summary(
                benchmark_id=self.config.benchmark_id,
                aggregate_metrics=result.aggregate_metrics,
                config=self.config.to_dict(),
            )

        finally:
            self.integrations.stop()

        # Finalize timing
        result.ended_at = datetime.utcnow()
        result.duration_seconds = time.time() - start_time

        # Print results
        self._print_results(result)

        # Export if configured
        if self.config.output_path:
            self._export_results(result)

        return result

    async def _run_all_episodes(
        self,
        seller_agent: Any,
        buyer_simulator: Callable,
    ) -> list[EpisodeResult]:
        """Run all episodes with controlled parallelism.

        Args:
            seller_agent: The seller agent.
            buyer_simulator: The buyer simulator.

        Returns:
            List of episode results.
        """
        semaphore = asyncio.Semaphore(self.config.parallelism)
        results = []

        async def run_with_semaphore(episode_index: int) -> EpisodeResult:
            async with semaphore:
                return await self._run_single_episode(
                    episode_index=episode_index,
                    seller_agent=seller_agent,
                    buyer_simulator=buyer_simulator,
                )

        # Create tasks for all episodes
        tasks = [run_with_semaphore(i) for i in range(self.config.num_episodes)]

        # Run with progress tracking
        for coro in asyncio.as_completed(tasks):
            ep_result = await coro
            results.append(ep_result)

            # Progress output
            status = "OK" if ep_result.succeeded else "FAIL"
            accepts = ep_result.total_accepts
            score = ep_result.final_score
            duration = ep_result.duration_seconds

            self.output_callback(
                f"Episode {ep_result.episode_index + 1}/{self.config.num_episodes} "
                f"(seed={ep_result.seed}): score={score:.2f}, accepts={accepts} "
                f"[{duration:.1f}s] [{status}]"
            )

        return results

    async def _run_single_episode(
        self,
        episode_index: int,
        seller_agent: Any,
        buyer_simulator: Callable,
    ) -> EpisodeResult:
        """Run a single episode asynchronously.

        Args:
            episode_index: Zero-based episode index.
            seller_agent: The seller agent.
            buyer_simulator: The buyer simulator.

        Returns:
            Episode result.
        """
        seed = self.config.get_episode_seed(episode_index)
        executor = EpisodeExecutor(self.config, self.integrations)

        # Run in executor to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            executor.run_episode,
            episode_index,
            seed,
            seller_agent,
            buyer_simulator,
            self._verbose_callback if self.config.verbose else None,
        )

    def _create_agents(self):
        """Create seller agent and buyer simulator.

        Returns:
            Tuple of (seller_agent, buyer_simulator, display_seller_name, display_buyer_name).
        """
        import os

        from salesbench.llm import DEFAULT_MODELS, detect_available_provider

        provider = detect_available_provider()
        if not provider:
            self.output_callback("ERROR: No LLM provider API key found.")
            self.output_callback("Set ONE of these environment variables:")
            self.output_callback("  - OPENAI_API_KEY")
            self.output_callback("  - ANTHROPIC_API_KEY")
            self.output_callback("  - OPENROUTER_API_KEY")
            return None, None, None, None

        from salesbench.agents.buyer_llm import create_buyer_simulator
        from salesbench.agents.seller_llm import LLMSellerAgent

        # Determine models
        seller_model = self.config.seller_model or os.environ.get("SALESBENCH_SELLER_MODEL")
        buyer_model = self.config.buyer_model or os.environ.get("SALESBENCH_BUYER_MODEL")
        buyer_temp = float(os.environ.get("SALESBENCH_BUYER_TEMPERATURE", "0.3"))

        # Create agents
        buyer_simulator = create_buyer_simulator(
            provider=provider,
            model=buyer_model,
            temperature=buyer_temp,
        )

        seller_agent = LLMSellerAgent(
            provider=provider,
            model=seller_model,
        )

        # Display names
        display_seller = seller_model or DEFAULT_MODELS.get(provider, "default")
        display_buyer = buyer_model or DEFAULT_MODELS.get(provider, "default")

        return (
            seller_agent,
            buyer_simulator,
            f"{display_seller} ({provider})",
            f"{display_buyer} ({provider})",
        )

    def _print_header(self) -> None:
        """Print benchmark header."""
        self.output_callback("")
        self.output_callback("SalesBench Benchmark Runner")
        self.output_callback("=" * 40)
        self.output_callback(f"Mode: {self.config.mode.value}")
        self.output_callback(f"Episodes: {self.config.num_episodes}")
        self.output_callback(f"Leads per episode: {self.config.num_leads}")
        self.output_callback(f"Parallelism: {self.config.parallelism}")
        self.output_callback(f"Benchmark ID: {self.config.benchmark_id}")

    def _print_results(self, result: BenchmarkResult) -> None:
        """Print benchmark results."""
        self.output_callback("")
        self.output_callback("Results")
        self.output_callback("=" * 40)

        metrics = result.aggregate_metrics
        self.output_callback(f"Total episodes:     {result.total_episodes}")
        self.output_callback(f"Completed:          {result.completed_episodes}")
        self.output_callback(f"Failed:             {result.failed_episodes}")
        self.output_callback("")

        if metrics:
            mean_score = metrics.get("mean_score", 0)
            std_score = metrics.get("std_score", 0)
            self.output_callback(f"Mean score:         {mean_score:.2f} (+/- {std_score:.2f})")

            acceptance_rate = metrics.get("mean_acceptance_rate", 0) * 100
            self.output_callback(f"Acceptance rate:    {acceptance_rate:.1f}%")

            success_rate = metrics.get("episode_success_rate", 0) * 100
            self.output_callback(f"Episode success:    {success_rate:.1f}%")

            pass_at_1 = metrics.get("pass_at_1", 0) * 100
            self.output_callback(f"Pass@1:             {pass_at_1:.1f}%")

            if metrics.get("pass_at_5"):
                pass_at_5 = metrics.get("pass_at_5", 0) * 100
                self.output_callback(f"Pass@5:             {pass_at_5:.1f}%")

            if metrics.get("pass_at_10"):
                pass_at_10 = metrics.get("pass_at_10", 0) * 100
                self.output_callback(f"Pass@10:            {pass_at_10:.1f}%")

            total_duration = metrics.get("total_duration_seconds", result.duration_seconds)
            self.output_callback(f"Total duration:     {total_duration:.1f}s")

        self.output_callback("")

        # Supabase info
        if self.integrations.supabase_enabled:
            self.output_callback(
                f"Results written to Supabase (benchmark_id: {self.config.benchmark_id})"
            )

        # Grafana info
        trace_url = self.integrations.get_grafana_trace_url(self.config.benchmark_id)
        if trace_url:
            self.output_callback(f"Traces: {trace_url}")

    def _export_results(self, result: BenchmarkResult) -> None:
        """Export results to JSON file."""
        if not self.config.output_path:
            return

        try:
            with open(self.config.output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            self.output_callback(f"Results exported to: {self.config.output_path}")
        except Exception as e:
            self.output_callback(f"Failed to export results: {e}")

    def _verbose_callback(self, message: str) -> None:
        """Callback for verbose output from episode executor."""
        if self.config.verbose:
            self.output_callback(message)


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Convenience function to run a benchmark.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult.
    """
    runner = BenchmarkRunner(config)
    return runner.run()


def run_test_benchmark(**kwargs) -> BenchmarkResult:
    """Run a quick test benchmark.

    Args:
        **kwargs: Override any BenchmarkConfig fields.

    Returns:
        BenchmarkResult.
    """
    config = BenchmarkConfig.from_mode(RunMode.TEST, **kwargs)
    return run_benchmark(config)


def run_production_benchmark(**kwargs) -> BenchmarkResult:
    """Run a full production benchmark.

    Args:
        **kwargs: Override any BenchmarkConfig fields.

    Returns:
        BenchmarkResult.
    """
    config = BenchmarkConfig.from_mode(RunMode.PRODUCTION, **kwargs)
    return run_benchmark(config)
