"""Benchmark configuration for SalesBench runner.

Defines run modes, presets, and configuration for benchmark execution.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from salesbench.core.config import BudgetConfig, SalesBenchConfig


class RunMode(str, Enum):
    """Benchmark run modes with different presets."""

    PRODUCTION = "production"  # 100 episodes, 100 leads, full integrations
    TEST = "test"  # 3 episodes, 5 leads, full integrations (quick validation)
    DEBUG = "debug"  # 1 episode, verbose output
    DEMO = "demo"  # 10 episodes, 50 leads, full integrations, verbose output


# Default presets for each mode
# Note: parallelism defaults to 1 for simplicity and reproducibility.
# Users can increase with --parallelism flag based on their API limits.
# (Following tau-bench pattern: https://github.com/sierra-research/tau-bench)
#
# safety_max_turns: Optional ceiling to prevent runaway episodes.
# None = no artificial limit (natural termination only via time/leads/quit).
# All modes default to None - users can configure via --safety-max-turns if needed.
#
# budget: Simulated time budget
# - total_hours: Total simulated hours for the episode
MODE_PRESETS = {
    RunMode.PRODUCTION: {
        "num_episodes": 100,
        "num_leads": 100,
        "safety_max_turns": None,  # Natural termination only
        "parallelism": 1,
        "budget": {
            "total_hours": 80,  # 10 days × 8 hours
        },
    },
    RunMode.TEST: {
        "num_episodes": 3,
        "num_leads": 5,
        "safety_max_turns": None,  # Natural termination only
        "parallelism": 1,
        "budget": {
            "total_hours": 16,  # 2 days × 8 hours
        },
    },
    RunMode.DEBUG: {
        "num_episodes": 1,
        "num_leads": 5,
        "safety_max_turns": None,  # Natural termination only
        "parallelism": 1,
        "budget": {
            "total_hours": 4,  # 1 day × 4 hours
        },
    },
    RunMode.DEMO: {
        "num_episodes": 3,
        "num_leads": 50,
        "safety_max_turns": None,  # Natural termination only
        "parallelism": 1,
        "budget": {
            "total_hours": 2,  # 1 day × 2 hours
        },
    },
}


def generate_benchmark_id() -> str:
    """Generate a unique benchmark ID."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"bench_{short_uuid}"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    This is the single source of truth for benchmark configuration.
    Episode-level settings (seed, num_leads, budget) are embedded in episode_config.

    Attributes:
        benchmark_id: Unique identifier for this benchmark run.
        name: User-provided name or description.
        num_episodes: Number of episodes to run.
        safety_max_turns: Optional safety ceiling for turns (None = no limit).
        parallelism: Number of concurrent episodes.
        seller_model: Model for seller agent (format: provider/model).
        buyer_model: Model for buyer simulator (format: provider/model).
        mode: Run mode (production, test, debug, demo).
        domain: Sales domain to benchmark (e.g., "insurance").
        enable_supabase: Whether to write to Supabase.
        enable_telemetry: Whether to send to OTel/Grafana.
        enable_json_storage: Whether to write results to JSON files.
        output_path: Optional path for JSON output.
        verbose: Enable verbose output.
        episode_config: Embedded episode configuration (seed, num_leads, budget).
    """

    benchmark_id: str = field(default_factory=generate_benchmark_id)
    name: str = ""
    num_episodes: int = 100
    safety_max_turns: Optional[int] = None  # None = natural termination only
    parallelism: int = 1
    seller_model: Optional[str] = None  # Format: provider/model (e.g., "openai/gpt-4o")
    buyer_model: Optional[str] = None  # Format: provider/model (e.g., "openai/gpt-4o-mini")
    mode: RunMode = RunMode.PRODUCTION
    domain: str = "insurance"
    enable_supabase: bool = False
    enable_telemetry: bool = False  # Disabled by default (requires OTEL collector)
    enable_json_storage: bool = True
    output_path: Optional[str] = None
    verbose: bool = False
    progress_interval: int = 5  # Update progress display every N turns (1 = every turn)
    # Embedded episode config - single source of truth for episode-level settings
    episode_config: SalesBenchConfig = field(default_factory=SalesBenchConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate parallelism
        if self.parallelism < 1:
            raise ValueError(f"parallelism must be >= 1, got {self.parallelism}")

        # Validate num_episodes
        if self.num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {self.num_episodes}")

        # Auto-generate name if not provided
        if not self.name:
            self.name = f"benchmark_{self.benchmark_id}"

    # Convenience accessors for backwards compatibility
    @property
    def base_seed(self) -> int:
        """Base seed for reproducibility (from episode_config)."""
        return self.episode_config.seed

    @property
    def num_leads(self) -> int:
        """Number of leads per episode (from episode_config)."""
        return self.episode_config.num_leads

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
        enable_supabase: bool = False,
        enable_telemetry: bool = True,
        enable_json_storage: bool = True,
        output_path: Optional[str] = None,
        verbose: bool = False,
        progress_interval: int = 5,
        # Override presets
        num_episodes: Optional[int] = None,
        num_leads: Optional[int] = None,
        safety_max_turns: Optional[int] = None,
        parallelism: Optional[int] = None,
        # Budget overrides
        total_hours: Optional[int] = None,
    ) -> "BenchmarkConfig":
        """Create a config from a run mode with optional overrides."""
        presets = MODE_PRESETS[mode]

        # For safety_max_turns, only use preset if not explicitly provided
        # None means "use preset default", but we need to distinguish from "explicitly set to None"
        final_safety_max_turns = presets["safety_max_turns"]
        if safety_max_turns is not None:
            final_safety_max_turns = safety_max_turns

        # Build BudgetConfig from preset with optional overrides
        budget_preset = presets.get("budget", {})
        budget_config = BudgetConfig(
            total_hours=total_hours if total_hours is not None else budget_preset.get("total_hours", 80),
        )

        # Build episode config with seed, num_leads, and budget
        episode_config = SalesBenchConfig(
            seed=base_seed,
            num_leads=num_leads if num_leads is not None else presets["num_leads"],
            budget=budget_config,
        )

        return cls(
            benchmark_id=benchmark_id or generate_benchmark_id(),
            name=name,
            num_episodes=num_episodes if num_episodes is not None else presets["num_episodes"],
            safety_max_turns=final_safety_max_turns,
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
            progress_interval=progress_interval,
            episode_config=episode_config,
        )

    @classmethod
    def from_cli_args(
        cls,
        mode: str = "production",
        episodes: Optional[int] = None,
        seed: int = 42,
        leads: Optional[int] = None,
        safety_max_turns: Optional[int] = None,
        parallelism: Optional[int] = None,
        seller_model: Optional[str] = None,
        buyer_model: Optional[str] = None,
        domain: str = "insurance",
        no_supabase: bool = False,
        enable_telemetry: bool = False,
        output: Optional[str] = None,
        verbose: bool = False,
        name: str = "",
        progress_interval: int = 5,
        # Budget overrides
        hours: Optional[int] = None,
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
            progress_interval=progress_interval,
            num_episodes=episodes,
            num_leads=leads,
            safety_max_turns=safety_max_turns,
            parallelism=parallelism,
            total_hours=hours,
        )

    def get_episode_seed(self, episode_index: int) -> int:
        """Get the seed for a specific episode.

        Args:
            episode_index: Zero-based episode index.

        Returns:
            Seed for the episode.
        """
        return self.episode_config.seed + episode_index

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "num_episodes": self.num_episodes,
            "base_seed": self.base_seed,  # Uses property for backwards compat
            "num_leads": self.num_leads,  # Uses property for backwards compat
            "safety_max_turns": self.safety_max_turns,
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
            "progress_interval": self.progress_interval,
            # Include full episode config for complete picture
            "episode_config": self.episode_config.to_dict(),
        }
