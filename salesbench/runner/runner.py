"""Main benchmark runner with parallel execution.

Orchestrates running multiple episodes in parallel with telemetry
and storage integrations.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from salesbench.runner.config import BenchmarkConfig, RunMode
from salesbench.runner.executor import EpisodeExecutor
from salesbench.runner.integrations import IntegrationManager
from salesbench.runner.results import BenchmarkResult, EpisodeProgress, EpisodeResult
from salesbench.storage.json_writer import JSONResultsWriter

logger = logging.getLogger(__name__)
console = Console()


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
                seller_trigger = seller_spec.config.compression_trigger
                self.output_callback(f"Seller context threshold: {seller_trigger:,} tokens")
            if buyer_spec and buyer_spec.config:
                buyer_trigger = buyer_spec.config.compression_trigger
                self.output_callback(f"Buyer context threshold: {buyer_trigger:,} tokens")

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
        """Run all episodes with controlled parallelism and real-time progress.

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
        budget = self.config.episode_config.budget

        # Track per-episode progress (thread-safe via lock)
        episode_progress: dict[int, EpisodeProgress] = {}
        progress_lock = threading.Lock()
        completed_episodes: list[tuple[int, EpisodeResult]] = []
        # Track last reported turn per episode for periodic verbose summaries
        last_verbose_turn: dict[int, int] = {}

        def on_progress(p: EpisodeProgress) -> None:
            """Thread-safe progress callback."""
            with progress_lock:
                episode_progress[p.episode_index] = p

                # In verbose mode, print periodic progress summaries
                if self.config.verbose:
                    last_turn = last_verbose_turn.get(p.episode_index, 0)
                    if p.turn >= last_turn + self.config.progress_interval or p.turn == 1:
                        last_verbose_turn[p.episode_index] = p.turn
                        resolved = p.leads_converted + p.leads_dnc
                        status = f"calling {p.current_lead_id}" if p.in_call else "between calls"
                        self.output_callback(
                            f"[Ep {p.episode_index + 1}] PROGRESS: "
                            f"turn {p.turn}, time {int(p.elapsed_hours)}h{int(p.elapsed_minutes):02d}m/{p.total_hours}h, "
                            f"leads {resolved}/{p.total_leads} ({p.leads_converted} converted, {p.leads_dnc} dnc), "
                            f"{status}"
                        )

        def build_progress_table() -> Table:
            """Build a table showing all active episode progress."""
            table = Table(box=None, show_header=False, padding=(0, 1))
            table.add_column("Episode", style="cyan", width=12)
            table.add_column("Time", width=30)
            table.add_column("Leads", width=40)
            table.add_column("Status", width=20)

            with progress_lock:
                if not episode_progress:
                    # Show placeholder while waiting for first progress
                    table.add_row(
                        "[dim]Starting...[/dim]",
                        "[dim]Waiting for episodes[/dim]",
                        "",
                        "[dim]Initializing[/dim]",
                    )
                    return table

                # Sort by episode index for consistent display
                for ep_idx in sorted(episode_progress.keys()):
                    p = episode_progress[ep_idx]

                    # Time progress bar
                    time_pct = p.time_progress
                    time_filled = int(time_pct * 20)
                    time_bar = "█" * time_filled + "░" * (20 - time_filled)
                    time_str = f"[yellow]{time_bar}[/yellow] {int(p.elapsed_hours)}h{int(p.elapsed_minutes):02d}m/{p.total_hours}h"

                    # Leads progress
                    resolved = p.leads_converted + p.leads_dnc
                    lead_pct = resolved / p.total_leads if p.total_leads > 0 else 0
                    lead_filled = int(lead_pct * 20)
                    lead_bar = "█" * lead_filled + "░" * (20 - lead_filled)
                    lead_str = (
                        f"[green]{lead_bar}[/green] "
                        f"{resolved}/{p.total_leads} "
                        f"([green]{p.leads_converted}[/green]✓ "
                        f"[red]{p.leads_dnc}[/red]✗)"
                    )

                    # Status
                    if p.in_call:
                        status = f"[blue]📞 {p.current_lead_id}[/blue]"
                    else:
                        status = f"[dim]Turn {p.turn}[/dim]"

                    table.add_row(
                        f"Ep {ep_idx + 1}/{total}",
                        time_str,
                        lead_str,
                        status,
                    )

            return table

        async def run_episodes_core():
            """Core episode execution logic."""
            if self.config.parallelism == 1:
                # Sequential execution
                for i in range(total):
                    ep_result = await self._run_single_episode(
                        episode_index=i,
                        seller_config=seller_config,
                        buyer_simulator=buyer_simulator,
                        seller_model_spec=seller_model_spec,
                        buyer_model_spec=buyer_model_spec,
                        progress_callback=on_progress,
                    )
                    results.append(ep_result)
                    with progress_lock:
                        episode_progress.pop(i, None)  # Remove completed
                    completed_episodes.append((i, ep_result))
            else:
                # Parallel execution with semaphore
                semaphore = asyncio.Semaphore(self.config.parallelism)

                async def run_with_semaphore(episode_index: int) -> tuple[int, EpisodeResult]:
                    async with semaphore:
                        result = await self._run_single_episode(
                            episode_index=episode_index,
                            seller_config=seller_config,
                            buyer_simulator=buyer_simulator,
                            seller_model_spec=seller_model_spec,
                            buyer_model_spec=buyer_model_spec,
                            progress_callback=on_progress,
                        )
                        with progress_lock:
                            episode_progress.pop(episode_index, None)
                        return (episode_index, result)

                # Create and run all tasks
                tasks = [run_with_semaphore(i) for i in range(total)]
                for coro in asyncio.as_completed(tasks):
                    ep_idx, ep_result = await coro
                    results.append(ep_result)
                    completed_episodes.append((ep_idx, ep_result))

        # Skip Live display when verbose mode is enabled to avoid output corruption
        if self.config.verbose:
            # Run without Live display - verbose output provides feedback
            await run_episodes_core()
        else:
            # Create Rich Live display for real-time updates
            with Live(
                Panel(build_progress_table(), title="Episode Progress", border_style="blue"),
                console=console,
                refresh_per_second=4,
                transient=True,
            ) as live:

                async def update_display():
                    """Periodically update the Live display."""
                    while True:
                        await asyncio.sleep(0.25)
                        live.update(
                            Panel(build_progress_table(), title="Episode Progress", border_style="blue")
                        )

                # Start display update task
                display_task = asyncio.create_task(update_display())

                try:
                    await run_episodes_core()
                finally:
                    display_task.cancel()
                    try:
                        await display_task
                    except asyncio.CancelledError:
                        pass

        # Print completion summary for each episode
        for ep_idx, ep_result in sorted(completed_episodes, key=lambda x: x[0]):
            self._print_episode_summary(ep_result)

        return results

    def _print_episode_summary(self, ep_result: EpisodeResult) -> None:
        """Print a completion summary for an episode."""
        status_color = "green" if ep_result.succeeded else "red"
        status_text = "✓" if ep_result.succeeded else "✗"
        accepts = ep_result.total_accepts
        score = ep_result.final_score
        duration = ep_result.duration_seconds
        total = self.config.num_episodes

        msg = (
            f"[{status_color}]{status_text}[/{status_color}] "
            f"Episode {ep_result.episode_index + 1}/{total} "
            f"(seed={ep_result.seed}): score={score:.2f}, accepts={accepts} "
            f"[dim][{duration:.1f}s][/dim]"
        )
        console.print(msg)

    async def _run_single_episode(
        self,
        episode_index: int,
        seller_config: dict,
        buyer_simulator: Callable,
        seller_model_spec=None,
        buyer_model_spec=None,
        progress_callback: Optional[Callable] = None,
    ) -> EpisodeResult:
        """Run a single episode asynchronously.

        Args:
            episode_index: Zero-based episode index.
            seller_config: Config dict to create seller agent.
            buyer_simulator: The buyer simulator (used as template for config).
            seller_model_spec: Model specification for seller.
            buyer_model_spec: Model specification for buyer.
            progress_callback: Optional callback for real-time progress updates.

        Returns:
            Episode result.
        """
        from salesbench.agents.buyer_llm import LLMBuyerSimulator
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

        # Create a fresh buyer simulator for this episode (avoid state sharing in parallel runs)
        # This ensures token tracking is accurate per-episode
        if isinstance(buyer_simulator, LLMBuyerSimulator):
            episode_buyer_simulator = LLMBuyerSimulator(
                provider=buyer_simulator.provider,
                model=buyer_simulator.model,
                temperature=buyer_simulator.temperature,
            )
        else:
            # Fallback for custom simulators - use as-is (user responsible for thread safety)
            episode_buyer_simulator = buyer_simulator

        # Create episode-specific verbose callback with episode prefix
        def episode_verbose_callback(message: str) -> None:
            if self._verbose_callback:
                prefix = f"[Ep {episode_index + 1}]"
                self._verbose_callback(f"{prefix} {message}")

        # Run in executor to not block event loop
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            executor.run_episode,
            episode_index,
            seed,
            seller_agent,
            episode_buyer_simulator,  # Use per-episode buyer simulator
            episode_verbose_callback if self.config.verbose else None,
            progress_callback,  # Pass through for real-time progress updates
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
        # Budget settings
        budget = self.config.episode_config.budget
        self.output_callback(f"Budget: {budget.total_hours} hours")
        if self.config.safety_max_turns:
            self.output_callback(f"Safety limit: {self.config.safety_max_turns} turns")
        else:
            self.output_callback("Safety limit: None (natural termination only)")
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

            # Episode success rate (at least 1 conversion)
            success_rate = metrics.get("episode_success_rate", 0) * 100
            self.output_callback(f"Episode success:    {success_rate:.1f}%")

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
