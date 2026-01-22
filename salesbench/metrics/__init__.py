"""Metrics and evaluation for SalesBench.

This module provides:
- Aggregation hooks for Prime Intellect integration
- Episode metrics collection
"""

from salesbench.metrics.collectors import (
    AggregateMetrics,
    EpisodeMetrics,
    MetricCollector,
)
from salesbench.metrics.hooks import (
    EpisodeMetricHook,
    MetricHook,
    PrimeIntellectHook,
)

__all__ = [
    # Hooks
    "MetricHook",
    "EpisodeMetricHook",
    "PrimeIntellectHook",
    # Collectors
    "MetricCollector",
    "EpisodeMetrics",
    "AggregateMetrics",
]
