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
# Note: parallelism defaults to 1 for simplicity and reproducibility.
# Users can increase with --parallelism flag based on their API limits.
# (Following tau-bench pattern: https://github.com/sierra-research/tau-bench)
MODE_PRESETS = {
    RunMode.PRODUCTION: {
        "num_episodes": 100,
        "num_leads": 100,
        "max_turns": 200,
        "parallelism": 1,
    },
    RunMode.TEST: {
        "num_episodes": 3,
        "num_leads": 5,
        "max_turns": 100,
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
        seller_model: Model for seller agent (format: provider/model).
        buyer_model: Model for buyer simulator (format: provider/model).
        mode: Run mode (production, test, debug).
        domain: Sales domain to benchmark (e.g., "insurance").
        enable_supabase: Whether to write to Supabase.
        enable_telemetry: Whether to send to OTel/Grafana.
        enable_json_storage: Whether to write results to JSON files.
        output_path: Optional path for JSON output.
        verbose: Enable verbose output.
    """

    benchmark_id: str = field(default_factory=generate_benchmark_id)
    name: str = ""
    num_episodes: int = 100
    base_seed: int = 42
    num_leads: int = 100
    max_turns: int = 200
    parallelism: int = 1
    seller_model: Optional[str] = None  # Format: provider/model (e.g., "openai/gpt-4o")
    buyer_model: Optional[str] = None  # Format: provider/model (e.g., "openai/gpt-4o-mini")
    mode: RunMode = RunMode.PRODUCTION
    domain: str = "insurance"
    enable_supabase: bool = True
    enable_telemetry: bool = False  # Disabled by default (requires OTEL collector)
    enable_json_storage: bool = True
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
        domain: str = "insurance",
        enable_supabase: bool = True,
        enable_telemetry: bool = True,
        enable_json_storage: bool = True,
        output_path: Optional[str] = None,
        verbose: bool = False,
        # Override presets
        num_episodes: Optional[int] = None,
        num_leads: Optional[int] = None,
        max_turns: Optional[int] = None,
        parallelism: Optional[int] = None,
    ) -> "BenchmarkConfig":
        """Create a config from a run mode with optional overrides."""
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
            domain=domain,
            enable_supabase=enable_supabase,
            enable_telemetry=enable_telemetry,
            enable_json_storage=enable_json_storage,
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
        domain: str = "insurance",
        no_supabase: bool = False,
        enable_telemetry: bool = False,
        output: Optional[str] = None,
        verbose: bool = False,
        name: str = "",
    ) -> "BenchmarkConfig":
        """Create config from CLI arguments."""
        run_mode = RunMode(mode)

        return cls.from_mode(
            mode=run_mode,
            name=name,
            base_seed=seed,
            seller_model=seller_model,
            buyer_model=buyer_model,
            domain=domain,
            enable_supabase=not no_supabase,
            enable_telemetry=enable_telemetry,
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
            "domain": self.domain,
            "enable_supabase": self.enable_supabase,
            "enable_telemetry": self.enable_telemetry,
            "enable_json_storage": self.enable_json_storage,
            "output_path": self.output_path,
            "verbose": self.verbose,
        }
