"""OpenTelemetry integration for SalesBench.

Provides distributed tracing and metrics export to Grafana/Prometheus.
This is an optional outer-layer integration kept separate from the
publishable environment package.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_tracer = None
_meter = None
_initialized = False


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry."""

    # Service identification
    service_name: str = "salesbench"
    service_version: str = "0.1.0"

    # OTLP exporter endpoints
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    )

    # Prometheus push gateway (for metrics)
    prometheus_gateway: str = field(
        default_factory=lambda: os.getenv("PROMETHEUS_PUSH_GATEWAY", "http://localhost:9091")
    )

    # Grafana configuration
    grafana_url: str = field(
        default_factory=lambda: os.getenv("GRAFANA_URL", "http://localhost:3000")
    )
    grafana_api_key: str = field(
        default_factory=lambda: os.getenv("GRAFANA_API_KEY", "")
    )

    # Sampling
    trace_sample_rate: float = 1.0  # Sample all traces by default

    # Export settings
    export_interval_millis: int = 5000
    max_export_batch_size: int = 512

    # Enable/disable
    enabled: bool = field(
        default_factory=lambda: os.getenv("OTEL_ENABLED", "false").lower() == "true"
    )

    def validate(self) -> bool:
        """Check if configuration is valid for telemetry."""
        return self.enabled and bool(self.otlp_endpoint)


class TelemetryManager:
    """Manages OpenTelemetry initialization and lifecycle."""

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig()
        self._provider = None
        self._meter_provider = None

    def init(self) -> bool:
        """Initialize OpenTelemetry."""
        global _tracer, _meter, _initialized

        if _initialized:
            return True

        if not self.config.enabled:
            logger.info("Telemetry disabled")
            return False

        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

            # Create resource
            resource = Resource.create({
                SERVICE_NAME: self.config.service_name,
                SERVICE_VERSION: self.config.service_version,
            })

            # Set up tracing
            self._provider = TracerProvider(resource=resource)
            span_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            self._provider.add_span_processor(
                BatchSpanProcessor(
                    span_exporter,
                    max_export_batch_size=self.config.max_export_batch_size,
                )
            )
            trace.set_tracer_provider(self._provider)
            _tracer = trace.get_tracer(self.config.service_name)

            # Set up metrics
            metric_exporter = OTLPMetricExporter(endpoint=self.config.otlp_endpoint)
            metric_reader = PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=self.config.export_interval_millis,
            )
            self._meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader],
            )
            metrics.set_meter_provider(self._meter_provider)
            _meter = metrics.get_meter(self.config.service_name)

            _initialized = True
            logger.info(f"Telemetry initialized with endpoint: {self.config.otlp_endpoint}")
            return True

        except ImportError as e:
            logger.warning(f"OpenTelemetry packages not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown OpenTelemetry providers."""
        global _initialized

        if self._provider:
            self._provider.shutdown()
        if self._meter_provider:
            self._meter_provider.shutdown()

        _initialized = False
        logger.info("Telemetry shutdown")

    def __enter__(self) -> "TelemetryManager":
        self.init()
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()


def init_telemetry(config: Optional[TelemetryConfig] = None) -> TelemetryManager:
    """Initialize telemetry with given config."""
    manager = TelemetryManager(config)
    manager.init()
    return manager


def get_tracer():
    """Get the global tracer."""
    global _tracer
    if _tracer is None:
        # Return a no-op tracer if not initialized
        try:
            from opentelemetry import trace
            return trace.get_tracer("salesbench")
        except ImportError:
            return NoOpTracer()
    return _tracer


def get_meter():
    """Get the global meter."""
    global _meter
    if _meter is None:
        # Return a no-op meter if not initialized
        try:
            from opentelemetry import metrics
            return metrics.get_meter("salesbench")
        except ImportError:
            return NoOpMeter()
    return _meter


class NoOpTracer:
    """No-op tracer for when OTel is not available."""

    def start_span(self, name: str, **kwargs):
        return NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs):
        return NoOpSpanContext()


class NoOpSpan:
    """No-op span."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self) -> None:
        pass


class NoOpSpanContext:
    """No-op context manager for spans."""

    def __enter__(self):
        return NoOpSpan()

    def __exit__(self, *args):
        pass


class NoOpMeter:
    """No-op meter for when OTel is not available."""

    def create_counter(self, name: str, **kwargs):
        return NoOpCounter()

    def create_histogram(self, name: str, **kwargs):
        return NoOpHistogram()

    def create_gauge(self, name: str, **kwargs):
        return NoOpGauge()


class NoOpCounter:
    """No-op counter."""

    def add(self, value: int, attributes: Optional[dict] = None) -> None:
        pass


class NoOpHistogram:
    """No-op histogram."""

    def record(self, value: float, attributes: Optional[dict] = None) -> None:
        pass


class NoOpGauge:
    """No-op gauge."""

    def set(self, value: float, attributes: Optional[dict] = None) -> None:
        pass
