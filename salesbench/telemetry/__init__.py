"""Telemetry layer for SalesBench.

Optional outer-layer integration for OpenTelemetry and Grafana.
"""

from salesbench.telemetry.otel import (
    TelemetryConfig,
    TelemetryManager,
    get_meter,
    get_tracer,
    init_telemetry,
)
from salesbench.telemetry.spans import (
    CallSpan,
    EpisodeSpan,
    SpanManager,
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
