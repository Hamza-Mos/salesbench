"""Benchmark runner module for SalesBench.

This module provides production-ready benchmark execution with:
- Parallel episode execution
- Supabase storage integration
- OpenTelemetry/Grafana tracing
- Flexible run modes (production, test, debug)

Example:
    from salesbench.runner import BenchmarkRunner, BenchmarkConfig, RunMode

    # Quick test run
    config = BenchmarkConfig.from_mode(RunMode.TEST)
    runner = BenchmarkRunner(config)
    result = runner.run()

    # Production run
    config = BenchmarkConfig.from_mode(
        RunMode.PRODUCTION,
        seller_model="gpt-4o",
        parallelism=10,
    )
    result = BenchmarkRunner(config).run()

CLI Usage:
    # Test mode (quick validation)
    salesbench run-benchmark --mode test

    # Production run
    salesbench run-benchmark --episodes 100 --parallelism 10

    # Custom configuration
    salesbench run-benchmark --episodes 50 --seller-model gpt-4o --output results.json
"""

from salesbench.runner.config import (
    MODE_PRESETS,
    BenchmarkConfig,
    RunMode,
    generate_benchmark_id,
)
from salesbench.runner.executor import (
    AsyncEpisodeExecutor,
    EpisodeExecutor,
)
from salesbench.runner.integrations import (
    IntegrationManager,
)
from salesbench.runner.results import (
    BenchmarkResult,
    EpisodeResult,
)
from salesbench.runner.runner import (
    BenchmarkRunner,
    run_benchmark,
    run_production_benchmark,
    run_test_benchmark,
)

__all__ = [
    # Config
    "BenchmarkConfig",
    "RunMode",
    "MODE_PRESETS",
    "generate_benchmark_id",
    # Results
    "EpisodeResult",
    "BenchmarkResult",
    # Integrations
    "IntegrationManager",
    # Executors
    "EpisodeExecutor",
    "AsyncEpisodeExecutor",
    # Runner
    "BenchmarkRunner",
    "run_benchmark",
    "run_test_benchmark",
    "run_production_benchmark",
]
