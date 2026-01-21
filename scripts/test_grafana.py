#!/usr/bin/env python3
"""Test Grafana Cloud telemetry connection.

Usage:
    python scripts/test_grafana.py

Requires .env file with:
    OTEL_ENABLED=true
    OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-prod-<region>.grafana.net/otlp
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64_credentials>
"""

import logging
import os
import sys
import time

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("Note: python-dotenv not installed, using existing environment variables")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def main():
    print("\n" + "=" * 60)
    print("GRAFANA CLOUD TELEMETRY TEST")
    print("=" * 60)

    # Check environment
    print("\n📋 Environment Variables:")
    env_vars = {
        "OTEL_ENABLED": os.getenv("OTEL_ENABLED", "not set"),
        "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "not set"),
        "GRAFANA_INSTANCE_ID": os.getenv("GRAFANA_INSTANCE_ID", "not set"),
        "GRAFANA_API_TOKEN": "✓ set" if os.getenv("GRAFANA_API_TOKEN") else "❌ not set",
    }
    for key, value in env_vars.items():
        print(f"   {key}: {value}")

    # Show OTLP headers with more detail (to debug encoding issues)
    headers_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
    if headers_env:
        print(f"   OTEL_EXPORTER_OTLP_HEADERS: ✓ set")
        # Show first 50 chars to check format
        preview = headers_env[:50] + "..." if len(headers_env) > 50 else headers_env
        print(f"      Preview: {preview}")
        if "%20" in headers_env:
            print("      ⚠️  WARNING: Contains '%20' (URL-encoded space)")
            print("         Your .env should have: Authorization=Basic <token>")
            print("         NOT: Authorization=Basic%20<token>")
    else:
        print(f"   OTEL_EXPORTER_OTLP_HEADERS: ❌ not set")

    # Check if enabled
    if os.getenv("OTEL_ENABLED", "").lower() != "true":
        print("\n❌ OTEL_ENABLED is not 'true'. Set it in your .env file.")
        sys.exit(1)

    # Import and initialize
    print("\n🔧 Initializing telemetry...")
    try:
        from salesbench.telemetry.otel import TelemetryConfig, TelemetryManager, get_tracer
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Make sure you're in the salesbench directory and it's installed.")
        sys.exit(1)

    config = TelemetryConfig()
    manager = TelemetryManager(config)

    if not manager.init():
        print("\n❌ Failed to initialize telemetry.")
        sys.exit(1)

    # Send test trace
    print("\n📤 Sending test trace...")
    tracer = get_tracer()

    test_id = f"test_{int(time.time())}"
    with tracer.start_as_current_span("grafana_connection_test") as span:
        span.set_attribute("test.id", test_id)
        span.set_attribute("test.timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        span.set_attribute("service.name", "salesbench")

        with tracer.start_as_current_span("child_operation") as child:
            child.set_attribute("operation", "test_child")
            time.sleep(0.05)

        span.add_event("test_completed", {"status": "success"})

    print(f"   Test ID: {test_id}")

    # Flush
    print("\n⏳ Flushing traces (waiting for export)...")
    if manager._provider:
        manager._provider.force_flush(timeout_millis=10000)

    # Done
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)
    print("\n📊 To view your trace in Grafana:")
    print("   1. Go to Grafana Cloud → Explore")
    print("   2. Select 'Tempo' as the data source")
    print("   3. Run query: {resource.service.name=\"salesbench\"}")
    print(f"   4. Look for trace: grafana_connection_test (test.id={test_id})")
    print("\n⏰ Note: Traces may take 30-60 seconds to appear.\n")

    manager.shutdown()


if __name__ == "__main__":
    main()
