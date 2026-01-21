"""OpenTelemetry integration for SalesBench.

Provides distributed tracing and metrics export to Grafana Cloud.
This is an optional outer-layer integration kept separate from the
publishable environment package.

Supports both local OTLP collectors and Grafana Cloud OTLP endpoints.

Configuration (set OTEL_ENABLED=true to enable):

Option 1 - Standard OpenTelemetry env vars (recommended):
  - OTEL_EXPORTER_OTLP_ENDPOINT: https://otlp-gateway-prod-<region>.grafana.net/otlp
  - OTEL_EXPORTER_OTLP_HEADERS: Authorization=Basic <base64(instance_id:api_token)>

Option 2 - Grafana-specific env vars:
  - GRAFANA_OTLP_ENDPOINT: https://otlp-gateway-prod-<region>.grafana.net/otlp
  - GRAFANA_INSTANCE_ID: Your Grafana Cloud instance ID
  - GRAFANA_API_TOKEN: API token with MetricsPublisher role
"""

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_tracer = None
_meter = None
_initialized = False


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry with Grafana Cloud support."""

    # Service identification
    service_name: str = "salesbench"
    service_version: str = "0.1.0"

    # OTLP exporter endpoint (local collector or Grafana Cloud)
    # For Grafana Cloud: https://otlp-gateway-prod-<region>.grafana.net/otlp
    otlp_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            os.getenv("GRAFANA_OTLP_ENDPOINT", "http://localhost:4317"),
        )
    )

    # Grafana Cloud authentication
    # Instance ID is your Grafana Cloud stack identifier (numeric)
    grafana_instance_id: str = field(
        default_factory=lambda: os.getenv("GRAFANA_INSTANCE_ID", "")
    )
    # API token with "MetricsPublisher" role for pushing telemetry
    grafana_api_token: str = field(
        default_factory=lambda: os.getenv("GRAFANA_API_TOKEN", "")
    )

    # Legacy Grafana configuration (for dashboard access, not telemetry push)
    grafana_url: str = field(
        default_factory=lambda: os.getenv("GRAFANA_URL", "http://localhost:3000")
    )
    grafana_api_key: str = field(default_factory=lambda: os.getenv("GRAFANA_API_KEY", ""))

    # Prometheus push gateway (for metrics, alternative to OTLP)
    prometheus_gateway: str = field(
        default_factory=lambda: os.getenv("PROMETHEUS_PUSH_GATEWAY", "http://localhost:9091")
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

    def is_grafana_cloud(self) -> bool:
        """Check if configured for Grafana Cloud (requires auth)."""
        return "grafana.net" in self.otlp_endpoint

    def get_otlp_headers(self) -> dict[str, str]:
        """Get authentication headers for OTLP exporter.

        Priority:
        1. OTEL_EXPORTER_OTLP_HEADERS env var (standard OTel format)
        2. GRAFANA_INSTANCE_ID + GRAFANA_API_TOKEN (Grafana Cloud Basic Auth)
        """
        from urllib.parse import unquote

        headers = {}

        # Check for standard OTEL headers env var first (takes precedence)
        # Format: "key1=value1,key2=value2" or "Authorization=Basic xxx"
        otel_headers_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        if otel_headers_env:
            # URL-decode the entire string first (handles %20 -> space, etc.)
            otel_headers_env = unquote(otel_headers_env)

            for pair in otel_headers_env.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    headers[key.strip()] = value.strip()
            return headers

        # Fall back to Grafana Cloud credentials
        if self.is_grafana_cloud() and self.grafana_instance_id and self.grafana_api_token:
            # Grafana Cloud uses Basic Auth: instance_id:api_token
            credentials = f"{self.grafana_instance_id}:{self.grafana_api_token}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        return headers


class TelemetryManager:
    """Manages OpenTelemetry initialization and lifecycle."""

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self.config = config or TelemetryConfig()
        self._provider = None
        self._meter_provider = None

    def init(self) -> bool:
        """Initialize OpenTelemetry with Grafana Cloud support."""
        global _tracer, _meter, _initialized

        if _initialized:
            logger.debug("Telemetry already initialized")
            return True

        if not self.config.enabled:
            logger.info("Telemetry disabled (set OTEL_ENABLED=true to enable)")
            return False

        # Debug: Log configuration
        logger.info("=" * 60)
        logger.info("TELEMETRY CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"  Service name: {self.config.service_name}")
        logger.info(f"  Service version: {self.config.service_version}")
        logger.info(f"  OTLP endpoint: {self.config.otlp_endpoint}")
        logger.info(f"  Is Grafana Cloud: {self.config.is_grafana_cloud()}")

        try:
            from opentelemetry import metrics, trace
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # Create resource with service identification
            resource = Resource.create(
                {
                    SERVICE_NAME: self.config.service_name,
                    SERVICE_VERSION: self.config.service_version,
                }
            )

            # Get auth headers for Grafana Cloud (empty dict for local collectors)
            headers = self.config.get_otlp_headers()

            # Debug: Log headers (mask the actual token but show format)
            if headers:
                for k, v in headers.items():
                    # Show first 30 chars to verify format (Basic xxx...)
                    masked = v[:30] + "..." if len(v) > 30 else v
                    logger.info(f"  Header '{k}': {masked}")
                    # Check for common issues
                    if k == "Authorization":
                        if v.startswith("Basic "):
                            logger.info("    ✓ Authorization header format looks correct")
                        elif v.startswith("Basic%20"):
                            logger.error("    ✗ Header has URL-encoded space (%20) - check .env file")
                        else:
                            logger.warning(f"    ? Unexpected format: starts with '{v[:10]}'")
            else:
                logger.warning("  Auth headers: NONE (this may cause 401 errors)")

            # Determine whether to use gRPC or HTTP based on endpoint
            # Grafana Cloud OTLP endpoint uses HTTP, local collectors typically use gRPC
            if self.config.is_grafana_cloud():
                # Use HTTP exporter for Grafana Cloud
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                # Grafana Cloud OTLP endpoints
                traces_endpoint = f"{self.config.otlp_endpoint}/v1/traces"
                metrics_endpoint = f"{self.config.otlp_endpoint}/v1/metrics"

                logger.info(f"  Traces endpoint: {traces_endpoint}")
                logger.info(f"  Metrics endpoint: {metrics_endpoint}")

                span_exporter = OTLPSpanExporter(
                    endpoint=traces_endpoint,
                    headers=headers,
                )
                metric_exporter = OTLPMetricExporter(
                    endpoint=metrics_endpoint,
                    headers=headers,
                )

                logger.info("  Exporter type: HTTP (Grafana Cloud)")
            else:
                # Use gRPC exporter for local OTLP collectors
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                span_exporter = OTLPSpanExporter(
                    endpoint=self.config.otlp_endpoint,
                    headers=headers if headers else None,
                )
                metric_exporter = OTLPMetricExporter(
                    endpoint=self.config.otlp_endpoint,
                    headers=headers if headers else None,
                )

                logger.info("  Exporter type: gRPC (local collector)")

            # Set up tracing with debug logging processor
            self._provider = TracerProvider(resource=resource)

            # Add debug span processor to log exports
            self._provider.add_span_processor(
                _DebugSpanProcessor(
                    BatchSpanProcessor(
                        span_exporter,
                        max_export_batch_size=self.config.max_export_batch_size,
                    )
                )
            )
            trace.set_tracer_provider(self._provider)
            _tracer = trace.get_tracer(self.config.service_name)

            # Set up metrics
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
            logger.info("=" * 60)
            logger.info("TELEMETRY INITIALIZED SUCCESSFULLY")
            logger.info("=" * 60)
            return True

        except ImportError as e:
            logger.error(f"OpenTelemetry packages not installed: {e}")
            logger.error(
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-http opentelemetry-exporter-otlp-proto-grpc"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}", exc_info=True)
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


class _DebugSpanProcessor:
    """Wrapper span processor that logs span events for debugging."""

    def __init__(self, wrapped_processor):
        self._wrapped = wrapped_processor
        self._span_count = 0

    def on_start(self, span, parent_context=None):
        self._span_count += 1
        logger.debug(f"[OTEL] Span started: {span.name} (total: {self._span_count})")
        if hasattr(self._wrapped, "on_start"):
            self._wrapped.on_start(span, parent_context)

    def on_end(self, span):
        logger.debug(
            f"[OTEL] Span ended: {span.name} "
            f"(duration: {(span.end_time - span.start_time) / 1e6:.2f}ms)"
        )
        self._wrapped.on_end(span)

    def shutdown(self):
        logger.info(f"[OTEL] Shutting down. Total spans created: {self._span_count}")
        self._wrapped.shutdown()

    def force_flush(self, timeout_millis=30000):
        logger.debug("[OTEL] Force flushing spans...")
        result = self._wrapped.force_flush(timeout_millis)
        logger.debug(f"[OTEL] Force flush result: {result}")
        return result


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


def test_telemetry_connection() -> bool:
    """Send a test trace to verify the connection to Grafana Cloud.

    Run this to verify your telemetry configuration:
        python -c "from salesbench.telemetry.otel import test_telemetry_connection; test_telemetry_connection()"

    Returns:
        True if test trace was sent successfully.
    """
    import time

    # Enable verbose logging
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("TELEMETRY CONNECTION TEST")
    print("=" * 60)

    # Check environment variables
    print("\nEnvironment variables:")
    print(f"  OTEL_ENABLED = {os.getenv('OTEL_ENABLED', 'not set')}")
    print(f"  OTEL_EXPORTER_OTLP_ENDPOINT = {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'not set')}")
    print(f"  OTEL_EXPORTER_OTLP_HEADERS = {'set (hidden)' if os.getenv('OTEL_EXPORTER_OTLP_HEADERS') else 'not set'}")
    print(f"  GRAFANA_OTLP_ENDPOINT = {os.getenv('GRAFANA_OTLP_ENDPOINT', 'not set')}")
    print(f"  GRAFANA_INSTANCE_ID = {os.getenv('GRAFANA_INSTANCE_ID', 'not set')}")
    print(f"  GRAFANA_API_TOKEN = {'set (hidden)' if os.getenv('GRAFANA_API_TOKEN') else 'not set'}")

    # Initialize telemetry
    print("\nInitializing telemetry...")
    config = TelemetryConfig()

    if not config.enabled:
        print("\n❌ OTEL_ENABLED is not set to 'true'. Telemetry is disabled.")
        print("   Set OTEL_ENABLED=true in your .env file.")
        return False

    manager = TelemetryManager(config)
    if not manager.init():
        print("\n❌ Failed to initialize telemetry. Check logs above.")
        return False

    # Send a test trace
    print("\nSending test trace...")
    tracer = get_tracer()

    with tracer.start_as_current_span("test_connection") as span:
        span.set_attribute("test.type", "connection_verification")
        span.set_attribute("test.timestamp", time.time())

        with tracer.start_as_current_span("test_child_span") as child:
            child.set_attribute("test.message", "Hello from SalesBench!")
            time.sleep(0.1)  # Small delay to make span visible

        span.add_event("test_event", {"message": "Connection test completed"})

    print("\n✓ Test trace created")

    # Force flush to ensure trace is sent
    print("\nFlushing traces to backend...")
    if manager._provider:
        manager._provider.force_flush(timeout_millis=10000)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Go to Grafana Cloud -> Explore -> Select 'Tempo'")
    print("2. Run query: {resource.service.name=\"salesbench\"}")
    print("3. You should see a 'test_connection' trace")
    print("\nNote: Traces may take 30-60 seconds to appear in Grafana.")

    # Clean shutdown
    manager.shutdown()

    return True


if __name__ == "__main__":
    # Run test when executed directly
    test_telemetry_connection()
