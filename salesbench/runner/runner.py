"""Main benchmark runner with parallel execution.

Orchestrates running multiple episodes in parallel with telemetry
and storage integrations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from salesbench.runner.config import BenchmarkConfig, RunMode
from salesbench.runner.executor import EpisodeExecutor
from salesbench.runner.integrations import IntegrationManager
from salesbench.runner.results import BenchmarkResult, EpisodeResult
from salesbench.storage.json_writer import JSONResultsWriter

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
        results_dir: str = "results",
    ):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration.
            output_callback: Optional callback for progress output.
            results_dir: Directory for JSON result files.
        """
        self.config = config
        self.output_callback = output_callback or (lambda x: print(x))
        self.integrations = IntegrationManager(config)
        self.json_writer = JSONResultsWriter(results_dir) if config.enable_json_storage else None

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
            # Create seller config and buyer simulator
            result_tuple = self._create_agents()
            (
                seller_config,
                buyer_simulator,
                display_seller,
                display_buyer,
                seller_spec,
                buyer_spec,
            ) = result_tuple

            if seller_config is None:
                self.output_callback("ERROR: Failed to create agents")
                return result

            self.output_callback(f"Seller model: {display_seller}")
            self.output_callback(f"Buyer model: {display_buyer}")

            # Log context window info if available
            if seller_spec and seller_spec.config:
                trigger = seller_spec.config.compression_trigger
                self.output_callback(f"Seller context threshold: {trigger:,} tokens")

            self.output_callback(
                f"Supabase: {'enabled' if self.integrations.supabase_enabled else 'disabled'}"
            )
            self.output_callback(
                f"Telemetry: {'enabled' if self.integrations.telemetry_enabled else 'disabled'}"
            )
            self.output_callback("")
            self.output_callback("Running episodes...")

            # Run episodes with parallelism (agents created per-episode)
            episode_results = await self._run_all_episodes(
                seller_config=seller_config,
                buyer_simulator=buyer_simulator,
                seller_model_spec=seller_spec,
                buyer_model_spec=buyer_spec,
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

        # Write to JSON storage
        if self.json_writer and self.config.enable_json_storage:
            # Extract custom name from output_path if provided (e.g., -o my_run)
            custom_name = None
            if self.config.output_path:
                custom_name = Path(self.config.output_path).stem

            results_dir = self.json_writer.write_benchmark(
                self.config.benchmark_id,
                result.to_dict(),
                include_traces=self.config.verbose,
                custom_name=custom_name,
            )
            self.output_callback(f"\nResults saved to: {results_dir}/")
            self.output_callback("  - summary.json (metrics, config, episode results)")
            if self.config.verbose:
                self.output_callback("  - traces.json (full conversation trajectories)")

        return result

    async def _run_all_episodes(
        self,
        seller_config: dict,
        buyer_simulator: Callable,
        seller_model_spec=None,
        buyer_model_spec=None,
    ) -> list[EpisodeResult]:
        """Run all episodes with controlled parallelism.

        Args:
            seller_config: Config dict to create seller agents (provider, model).
            buyer_simulator: The buyer simulator.
            seller_model_spec: Model specification for seller.
            buyer_model_spec: Model specification for buyer.

        Returns:
            List of episode results.
        """
        results = []
        total = self.config.num_episodes

        if self.config.parallelism == 1:
            # Sequential execution - guarantees order
            for i in range(total):
                ep_result = await self._run_single_episode(
                    episode_index=i,
                    seller_config=seller_config,
                    buyer_simulator=buyer_simulator,
                    seller_model_spec=seller_model_spec,
                    buyer_model_spec=buyer_model_spec,
                )
                results.append(ep_result)
                self._print_episode_progress(ep_result, completed_count=None)
        else:
            # Parallel execution with semaphore
            semaphore = asyncio.Semaphore(self.config.parallelism)
            completed = 0

            async def run_with_semaphore(episode_index: int) -> EpisodeResult:
                async with semaphore:
                    return await self._run_single_episode(
                        episode_index=episode_index,
                        seller_config=seller_config,
                        buyer_simulator=buyer_simulator,
                        seller_model_spec=seller_model_spec,
                        buyer_model_spec=buyer_model_spec,
                    )

            # Create tasks for all episodes
            tasks = [run_with_semaphore(i) for i in range(total)]

            # Run with progress tracking (as_completed yields in completion order)
            for coro in asyncio.as_completed(tasks):
                ep_result = await coro
                results.append(ep_result)
                completed += 1
                self._print_episode_progress(ep_result, completed_count=completed)

        return results

    def _print_episode_progress(
        self, ep_result: EpisodeResult, completed_count: int | None
    ) -> None:
        """Print progress for a completed episode.

        Args:
            ep_result: The episode result.
            completed_count: If provided, shows "Completed X/Y" format for parallel mode.
                           If None, shows "Episode X/Y" format for sequential mode.
        """
        status = "OK" if ep_result.succeeded else "FAIL"
        accepts = ep_result.total_accepts
        score = ep_result.final_score
        duration = ep_result.duration_seconds
        total = self.config.num_episodes

        if completed_count is None:
            # Sequential mode: Episode 1/100 (seed=42): ...
            self.output_callback(
                f"Episode {ep_result.episode_index + 1}/{total} "
                f"(seed={ep_result.seed}): score={score:.2f}, accepts={accepts} "
                f"[{duration:.1f}s] [{status}]"
            )
        else:
            # Parallel mode: Completed 1/100 [Episode 59] (seed=100): ...
            self.output_callback(
                f"Completed {completed_count}/{total} [Episode {ep_result.episode_index + 1}] "
                f"(seed={ep_result.seed}): score={score:.2f}, accepts={accepts} "
                f"[{duration:.1f}s] [{status}]"
            )

    async def _run_single_episode(
        self,
        episode_index: int,
        seller_config: dict,
        buyer_simulator: Callable,
        seller_model_spec=None,
        buyer_model_spec=None,
    ) -> EpisodeResult:
        """Run a single episode asynchronously.

        Args:
            episode_index: Zero-based episode index.
            seller_config: Config dict to create seller agent.
            buyer_simulator: The buyer simulator.
            seller_model_spec: Model specification for seller.
            buyer_model_spec: Model specification for buyer.

        Returns:
            Episode result.
        """
        from salesbench.agents.seller_llm import LLMSellerAgent

        seed = self.config.get_episode_seed(episode_index)
        executor = EpisodeExecutor(
            self.config,
            self.integrations,
            seller_model_spec=seller_model_spec,
            buyer_model_spec=buyer_model_spec,
        )

        # Create a fresh agent for this episode (avoid state sharing in parallel runs)
        seller_agent = LLMSellerAgent(
            provider=seller_config["provider"],
            model=seller_config["model"],
        )

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
        """Create seller config and buyer simulator.

        Parses model specs in provider/model format (e.g., "openai/gpt-4o")
        and routes to the correct API based on the provider.

        Returns:
            Tuple of (seller_config, buyer_simulator, display_seller_name, display_buyer_name,
                      seller_model_spec, buyer_model_spec).
            seller_config is a dict with provider/model to create agents per-episode.
        """
        import os

        from salesbench.agents.buyer_llm import create_buyer_simulator
        from salesbench.llm import get_api_key_for_provider
        from salesbench.models import DEFAULT_BUYER_MODEL, parse_model_spec

        # Parse seller model (e.g., "openai/gpt-4o" -> provider="openai", model="gpt-4o")
        seller_model_str = self.config.seller_model or os.environ.get("SALESBENCH_SELLER_MODEL")
        if not seller_model_str:
            self.output_callback("ERROR: No seller model specified.")
            self.output_callback("Use --models provider/model (e.g., --models openai/gpt-4o)")
            return None, None, None, None, None, None

        try:
            seller_spec = parse_model_spec(seller_model_str)
        except ValueError as e:
            self.output_callback(f"ERROR: {e}")
            return None, None, None, None, None, None

        # Parse buyer model
        buyer_model_str = (
            self.config.buyer_model
            or os.environ.get("SALESBENCH_BUYER_MODEL")
            or DEFAULT_BUYER_MODEL
        )
        try:
            buyer_spec = parse_model_spec(buyer_model_str)
        except ValueError as e:
            self.output_callback(f"ERROR: {e}")
            return None, None, None, None, None, None

        # Check API keys for both providers
        seller_api_key = get_api_key_for_provider(seller_spec.provider)
        if not seller_api_key:
            self.output_callback(
                f"ERROR: No API key found for seller provider '{seller_spec.provider}'."
            )
            self.output_callback(
                f"Set the appropriate environment variable for {seller_spec.provider}."
            )
            return None, None, None, None, None, None

        buyer_api_key = get_api_key_for_provider(buyer_spec.provider)
        if not buyer_api_key:
            self.output_callback(
                f"ERROR: No API key found for buyer provider '{buyer_spec.provider}'."
            )
            self.output_callback(
                f"Set the appropriate environment variable for {buyer_spec.provider}."
            )
            return None, None, None, None, None, None

        buyer_temp = float(os.environ.get("SALESBENCH_BUYER_TEMPERATURE", "0.0"))

        # Create buyer simulator (stateless, safe to share)
        buyer_simulator = create_buyer_simulator(
            provider=buyer_spec.provider,
            model=buyer_spec.model,
            temperature=buyer_temp,
        )

        # Return seller config (agents created per-episode to avoid state sharing)
        seller_config = {
            "provider": seller_spec.provider,
            "model": seller_spec.model,
        }

        return (
            seller_config,
            buyer_simulator,
            str(seller_spec),
            str(buyer_spec),
            seller_spec,
            buyer_spec,
        )

    def _print_header(self) -> None:
        """Print benchmark header."""
        self.output_callback("")
        self.output_callback("SalesBench Benchmark Runner")
        self.output_callback("=" * 40)
        self.output_callback(f"Mode: {self.config.mode.value}")
        self.output_callback(f"Domain: {self.config.domain}")
        self.output_callback(f"Episodes: {self.config.num_episodes}")
        self.output_callback(f"Leads per episode: {self.config.num_leads}")
        self.output_callback(f"Max turns per episode: {self.config.max_turns}")
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

            # Token usage
            token_usage = metrics.get("total_token_usage", {})
            if token_usage:
                self.output_callback("")
                self.output_callback("Token Usage (all episodes):")
                seller_in = token_usage.get("seller_input_tokens", 0)
                seller_out = token_usage.get("seller_output_tokens", 0)
                buyer_in = token_usage.get("buyer_input_tokens", 0)
                buyer_out = token_usage.get("buyer_output_tokens", 0)
                self.output_callback(f"  Seller: {seller_in:,} input, {seller_out:,} output")
                self.output_callback(f"  Buyer:  {buyer_in:,} input, {buyer_out:,} output")
                self.output_callback(
                    f"  Total:  {seller_in + buyer_in:,} input, {seller_out + buyer_out:,} output"
                )

            # Cost breakdown
            cost_breakdown = metrics.get("total_cost_breakdown", {})
            if cost_breakdown:
                self.output_callback("")
                self.output_callback("Cost Breakdown (all episodes):")
                seller_in_cost = cost_breakdown.get("seller_input_cost", 0)
                seller_out_cost = cost_breakdown.get("seller_output_cost", 0)
                buyer_in_cost = cost_breakdown.get("buyer_input_cost", 0)
                buyer_out_cost = cost_breakdown.get("buyer_output_cost", 0)
                total_cost = cost_breakdown.get("total_cost", 0)
                seller_available = cost_breakdown.get("seller_pricing_available", False)
                buyer_available = cost_breakdown.get("buyer_pricing_available", False)

                if seller_available:
                    self.output_callback(
                        f"  Seller: ${seller_in_cost:.4f} input, ${seller_out_cost:.4f} output"
                    )
                else:
                    self.output_callback("  Seller: N/A (pricing not available)")

                if buyer_available:
                    self.output_callback(
                        f"  Buyer:  ${buyer_in_cost:.4f} input, ${buyer_out_cost:.4f} output"
                    )
                else:
                    self.output_callback("  Buyer:  N/A (pricing not available)")

                if seller_available and buyer_available:
                    total_in = seller_in_cost + buyer_in_cost
                    total_out = seller_out_cost + buyer_out_cost
                    self.output_callback(
                        f"  Total:  ${total_in:.4f} input, ${total_out:.4f} output"
                    )
                    self.output_callback(f"  Total Cost: ${total_cost:.4f}")
                elif seller_available or buyer_available:
                    self.output_callback(f"  Partial Total Cost: ${total_cost:.4f}")

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
