"""Benchmark configuration for SalesBench runner.

Defines run modes, presets, and configuration for benchmark execution.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RunMode(str, Enum):
    """Benchmark run modes with different presets."""

    PRODUCTION = "production"  # 100 episodes, 100 leads, full integrations
    TEST = "test"  # 3 episodes, 5 leads, full integrations (quick validation)
    DEBUG = "debug"  # 1 episode, verbose output


# Default presets for each mode
MODE_PRESETS = {
    RunMode.PRODUCTION: {
        "num_episodes": 100,
        "num_leads": 100,
        "max_turns": 200,
        "parallelism": 5,
    },
    RunMode.TEST: {
        "num_episodes": 3,
        "num_leads": 5,
        "max_turns": 30,
        "parallelism": 1,
    },
    RunMode.DEBUG: {
        "num_episodes": 1,
        "num_leads": 5,
        "max_turns": 50,
        "parallelism": 1,
    },
}


def generate_benchmark_id() -> str:
    """Generate a unique benchmark ID."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"bench_{short_uuid}"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        benchmark_id: Unique identifier for this benchmark run.
        name: User-provided name or description.
        num_episodes: Number of episodes to run.
        base_seed: Starting seed for reproducibility.
        num_leads: Number of leads per episode.
        max_turns: Maximum turns per episode.
        parallelism: Number of concurrent episodes.
        seller_model: Model to use for the seller agent.
        buyer_model: Model to use for the buyer simulator.
        mode: Run mode (production, test, debug).
        enable_supabase: Whether to write to Supabase.
        enable_telemetry: Whether to send to OTel/Grafana.
        output_path: Optional path for JSON output.
        verbose: Enable verbose output.
    """

    benchmark_id: str = field(default_factory=generate_benchmark_id)
    name: str = ""
    num_episodes: int = 100
    base_seed: int = 42
    num_leads: int = 100
    max_turns: int = 200
    parallelism: int = 5
    seller_model: Optional[str] = None
    buyer_model: Optional[str] = None
    mode: RunMode = RunMode.PRODUCTION
    enable_supabase: bool = True
    enable_telemetry: bool = True
    output_path: Optional[str] = None
    verbose: bool = False

    def __post_init__(self):
        """Apply mode presets if not explicitly overridden."""
        # Auto-generate name if not provided
        if not self.name:
            self.name = f"benchmark_{self.benchmark_id}"

    @classmethod
    def from_mode(
        cls,
        mode: RunMode,
        *,
        benchmark_id: Optional[str] = None,
        name: str = "",
        base_seed: int = 42,
        seller_model: Optional[str] = None,
        buyer_model: Optional[str] = None,
        enable_supabase: bool = True,
        enable_telemetry: bool = True,
        output_path: Optional[str] = None,
        verbose: bool = False,
        # Override presets
        num_episodes: Optional[int] = None,
        num_leads: Optional[int] = None,
        max_turns: Optional[int] = None,
        parallelism: Optional[int] = None,
    ) -> "BenchmarkConfig":
        """Create a config from a run mode with optional overrides.

        Args:
            mode: The run mode to use as base.
            benchmark_id: Optional custom benchmark ID.
            name: Name for the benchmark run.
            base_seed: Starting seed for reproducibility.
            seller_model: Model for seller agent.
            buyer_model: Model for buyer simulator.
            enable_supabase: Whether to write to Supabase.
            enable_telemetry: Whether to send telemetry.
            output_path: Path for JSON output.
            verbose: Enable verbose output.
            num_episodes: Override preset episode count.
            num_leads: Override preset lead count.
            max_turns: Override preset max turns.
            parallelism: Override preset parallelism.

        Returns:
            Configured BenchmarkConfig.
        """
        presets = MODE_PRESETS[mode]

        return cls(
            benchmark_id=benchmark_id or generate_benchmark_id(),
            name=name,
            num_episodes=num_episodes if num_episodes is not None else presets["num_episodes"],
            base_seed=base_seed,
            num_leads=num_leads if num_leads is not None else presets["num_leads"],
            max_turns=max_turns if max_turns is not None else presets["max_turns"],
            parallelism=parallelism if parallelism is not None else presets["parallelism"],
            seller_model=seller_model,
            buyer_model=buyer_model,
            mode=mode,
            enable_supabase=enable_supabase,
            enable_telemetry=enable_telemetry,
            output_path=output_path,
            verbose=verbose,
        )

    @classmethod
    def from_cli_args(
        cls,
        mode: str = "production",
        episodes: Optional[int] = None,
        seed: int = 42,
        leads: Optional[int] = None,
        max_turns: Optional[int] = None,
        parallelism: Optional[int] = None,
        seller_model: Optional[str] = None,
        buyer_model: Optional[str] = None,
        no_supabase: bool = False,
        no_telemetry: bool = False,
        output: Optional[str] = None,
        verbose: bool = False,
        name: str = "",
    ) -> "BenchmarkConfig":
        """Create config from CLI arguments.

        Args:
            mode: Run mode string.
            episodes: Number of episodes.
            seed: Base random seed.
            leads: Leads per episode.
            max_turns: Max turns per episode.
            parallelism: Concurrent episodes.
            seller_model: Seller model name.
            buyer_model: Buyer model name.
            no_supabase: Disable Supabase.
            no_telemetry: Disable telemetry.
            output: Output file path.
            verbose: Verbose output.
            name: Benchmark name.

        Returns:
            Configured BenchmarkConfig.
        """
        run_mode = RunMode(mode)

        return cls.from_mode(
            mode=run_mode,
            name=name,
            base_seed=seed,
            seller_model=seller_model,
            buyer_model=buyer_model,
            enable_supabase=not no_supabase,
            enable_telemetry=not no_telemetry,
            output_path=output,
            verbose=verbose,
            num_episodes=episodes,
            num_leads=leads,
            max_turns=max_turns,
            parallelism=parallelism,
        )

    def get_episode_seed(self, episode_index: int) -> int:
        """Get the seed for a specific episode.

        Args:
            episode_index: Zero-based episode index.

        Returns:
            Seed for the episode.
        """
        return self.base_seed + episode_index

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "num_episodes": self.num_episodes,
            "base_seed": self.base_seed,
            "num_leads": self.num_leads,
            "max_turns": self.max_turns,
            "parallelism": self.parallelism,
            "seller_model": self.seller_model,
            "buyer_model": self.buyer_model,
            "mode": self.mode.value,
            "enable_supabase": self.enable_supabase,
            "enable_telemetry": self.enable_telemetry,
            "output_path": self.output_path,
            "verbose": self.verbose,
        }
