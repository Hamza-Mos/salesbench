"""Telemetry utilities for CLI commands.

Provides a clean context manager for telemetry initialization and shutdown.
"""

from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def telemetry_context(verbose: bool = False) -> Iterator[Optional["TelemetryManager"]]:
    """Context manager for telemetry initialization and cleanup.

    Args:
        verbose: Whether to print telemetry status messages.

    Yields:
        TelemetryManager instance if enabled and initialized, None otherwise.

    Example:
        with telemetry_context(verbose=True) as telemetry:
            if telemetry:
                tracer = telemetry.get_tracer()
                # ... use tracer
        # Telemetry automatically cleaned up
    """
    telemetry_manager = None

    try:
        from salesbench.telemetry.otel import TelemetryConfig, TelemetryManager

        telemetry_config = TelemetryConfig()
        if telemetry_config.enabled:
            telemetry_manager = TelemetryManager(telemetry_config)
            if telemetry_manager.init():
                if verbose:
                    print("Telemetry: enabled (traces will be sent to Grafana)")
            else:
                if verbose:
                    print("Telemetry: failed to initialize")
                telemetry_manager = None
        else:
            if verbose:
                print("Telemetry: disabled (set OTEL_ENABLED=true to enable)")
    except ImportError:
        if verbose:
            print("Telemetry: not available (missing opentelemetry packages)")

    try:
        yield telemetry_manager
    finally:
        if telemetry_manager:
            if verbose:
                print("\nFlushing telemetry traces...")
            telemetry_manager.shutdown()
            if verbose:
                print("Telemetry: traces sent to Grafana")


def get_tracer(telemetry_manager: Optional["TelemetryManager"]):
    """Get tracer from telemetry manager if available.

    Args:
        telemetry_manager: Optional telemetry manager instance.

    Returns:
        Tracer instance or None.
    """
    if telemetry_manager is None:
        return None

    try:
        from salesbench.telemetry.otel import get_tracer
        return get_tracer()
    except ImportError:
        return None


def get_span_manager(telemetry_manager: Optional["TelemetryManager"]):
    """Get span manager from telemetry if available.

    Args:
        telemetry_manager: Optional telemetry manager instance.

    Returns:
        SpanManager instance or None.
    """
    if telemetry_manager is None:
        return None

    try:
        from salesbench.telemetry.spans import get_span_manager
        return get_span_manager()
    except ImportError:
        return None
