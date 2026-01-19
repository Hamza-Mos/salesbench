"""Metrics and evaluation for SalesBench.

This module provides:
- pass@k computation for benchmark evaluation
- Aggregation hooks for Prime Intellect integration
- Episode metrics collection
"""

from salesbench.metrics.pass_at_k import (
    PassAtKComputer,
    PassAtKConfig,
    PassAtKResult,
    compute_pass_at_k,
    estimate_pass_at_k,
)
from salesbench.metrics.hooks import (
    MetricHook,
    EpisodeMetricHook,
    PassAtKHook,
    PrimeIntellectHook,
)
from salesbench.metrics.collectors import (
    MetricCollector,
    EpisodeMetrics,
    AggregateMetrics,
)

__all__ = [
    # pass@k
    "PassAtKComputer",
    "PassAtKConfig",
    "PassAtKResult",
    "compute_pass_at_k",
    "estimate_pass_at_k",
    # Hooks
    "MetricHook",
    "EpisodeMetricHook",
    "PassAtKHook",
    "PrimeIntellectHook",
    # Collectors
    "MetricCollector",
    "EpisodeMetrics",
    "AggregateMetrics",
]
