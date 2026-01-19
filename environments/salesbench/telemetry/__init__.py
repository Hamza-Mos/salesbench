"""Telemetry layer for SalesBench.

Optional outer-layer integration for OpenTelemetry and Grafana.
"""

from salesbench.telemetry.otel import (
    TelemetryConfig,
    TelemetryManager,
    init_telemetry,
    get_tracer,
    get_meter,
)
from salesbench.telemetry.spans import (
    SpanManager,
    EpisodeSpan,
    CallSpan,
    ToolSpan,
)

__all__ = [
    "TelemetryConfig",
    "TelemetryManager",
    "init_telemetry",
    "get_tracer",
    "get_meter",
    "SpanManager",
    "EpisodeSpan",
    "CallSpan",
    "ToolSpan",
]
