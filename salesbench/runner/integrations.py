"""Integration manager for benchmark runs.

Wires together TelemetryManager and SupabaseWriter with graceful
degradation when integrations are disabled or unavailable.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from salesbench.runner.config import BenchmarkConfig
from salesbench.runner.results import EpisodeResult

logger = logging.getLogger(__name__)


class NoOpSpanManager:
    """No-op span manager for when telemetry is disabled."""

    class NoOpEpisodeSpan:
        """No-op episode span."""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_final_score(self, score: float) -> None:
            pass

        def set_metrics(self, metrics: dict) -> None:
            pass

        def record_accept(self, plan_id: str, premium: float) -> None:
            pass

        def record_reject(self, plan_id: str) -> None:
            pass

        def record_dnc_violation(self, lead_id: str) -> None:
            pass

        def finish(self, final_score: float, metrics: dict) -> None:
            pass

    def episode_span(self, *args, **kwargs):
        """Return a no-op episode span context manager."""
        return self.NoOpEpisodeSpan()


class IntegrationManager:
    """Manages integrations (telemetry, storage) for benchmark runs.

    Provides a unified interface for:
    - OpenTelemetry tracing (via TelemetryManager)
    - Supabase storage (via SupabaseWriter)

    Both integrations gracefully degrade if disabled or unavailable.

    Example:
        config = BenchmarkConfig(enable_supabase=True, enable_telemetry=True)
        integrations = IntegrationManager(config)

        integrations.start()
        try:
            span = integrations.get_span_manager()
            with span.episode_span(...) as episode:
                # Run episode
                pass
            integrations.write_episode_result(result)
        finally:
            integrations.stop()
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize the integration manager.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self._telemetry_manager = None
        self._supabase_writer = None
        self._span_manager = None
        self._started = False

    @property
    def telemetry_enabled(self) -> bool:
        """Check if telemetry is actually enabled and initialized."""
        return self._telemetry_manager is not None

    @property
    def supabase_enabled(self) -> bool:
        """Check if Supabase is actually enabled and initialized."""
        return self._supabase_writer is not None and self._supabase_writer.enabled

    def start(self) -> None:
        """Start all enabled integrations."""
        if self._started:
            return

        # Initialize telemetry
        if self.config.enable_telemetry:
            try:
                from salesbench.telemetry.otel import TelemetryConfig, TelemetryManager
                from salesbench.telemetry.spans import get_span_manager

                telemetry_config = TelemetryConfig(enabled=True)
                self._telemetry_manager = TelemetryManager(telemetry_config)

                if self._telemetry_manager.init():
                    self._span_manager = get_span_manager()
                    logger.info("Telemetry initialized")
                else:
                    logger.info("Telemetry disabled (OTEL_ENABLED not set)")
                    self._telemetry_manager = None

            except ImportError as e:
                logger.warning(f"Telemetry packages not available: {e}")
                self._telemetry_manager = None
            except Exception as e:
                logger.warning(f"Failed to initialize telemetry: {e}")
                self._telemetry_manager = None

        # Initialize Supabase
        if self.config.enable_supabase:
            try:
                from salesbench.storage.supabase_writer import SupabaseConfig, SupabaseWriter

                supabase_config = SupabaseConfig()
                if supabase_config.validate():
                    self._supabase_writer = SupabaseWriter(supabase_config)
                    self._supabase_writer.start()
                    logger.info("Supabase writer started")
                else:
                    logger.info("Supabase disabled (SUPABASE_URL/KEY not set)")
                    self._supabase_writer = None

            except ImportError as e:
                logger.warning(f"Supabase package not available: {e}")
                self._supabase_writer = None
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase: {e}")
                self._supabase_writer = None

        self._started = True

    def stop(self) -> None:
        """Stop all integrations and flush pending data."""
        if not self._started:
            return

        # Stop Supabase (flushes pending writes)
        if self._supabase_writer:
            try:
                self._supabase_writer.stop()
                logger.info("Supabase writer stopped")
            except Exception as e:
                logger.error(f"Error stopping Supabase writer: {e}")

        # Shutdown telemetry
        if self._telemetry_manager:
            try:
                self._telemetry_manager.shutdown()
                logger.info("Telemetry shutdown")
            except Exception as e:
                logger.error(f"Error shutting down telemetry: {e}")

        self._started = False

    def get_span_manager(self):
        """Get the span manager for creating telemetry spans.

        Returns:
            SpanManager if telemetry is enabled, otherwise NoOpSpanManager.
        """
        if self._span_manager:
            return self._span_manager
        return NoOpSpanManager()

    def write_episode_start(
        self,
        episode_id: str,
        benchmark_id: str,
        seed: int,
        model_name: str,
        num_leads: int,
        config_dict: Optional[dict] = None,
    ) -> None:
        """Write episode start record to Supabase.

        Args:
            episode_id: Unique episode identifier.
            benchmark_id: Parent benchmark ID.
            seed: Random seed for episode.
            model_name: Model being used.
            num_leads: Number of leads.
            config_dict: Optional config to store.
        """
        if not self._supabase_writer:
            return

        try:
            from salesbench.storage.supabase_writer import EpisodeRecord

            record = EpisodeRecord(
                episode_id=episode_id,
                seed=seed,
                model_name=model_name,
                started_at=datetime.utcnow(),
                num_leads=num_leads,
                config={
                    "benchmark_id": benchmark_id,
                    **(config_dict or {}),
                },
            )
            self._supabase_writer.write_episode(record)

        except Exception as e:
            logger.warning(f"Failed to write episode start: {e}")

    def write_episode_end(
        self,
        episode_id: str,
        final_score: float,
        metrics: dict,
        model_name: str,
    ) -> None:
        """Write episode completion metrics to Supabase.

        Since the batch writer uses insert, we write metrics as separate records.

        Args:
            episode_id: Episode identifier.
            final_score: Final episode score.
            metrics: Episode metrics dict.
            model_name: Model name for tagging.
        """
        if not self._supabase_writer:
            return

        try:
            from salesbench.storage.supabase_writer import MetricRecord

            # Write key metrics as individual records
            metric_names = [
                ("final_score", final_score),
                ("total_accepts", metrics.get("accepted_offers", 0)),
                ("total_rejects", metrics.get("rejected_offers", 0)),
                ("total_calls", metrics.get("total_calls", 0)),
                ("dnc_violations", metrics.get("dnc_violations", 0)),
                ("acceptance_rate", metrics.get("acceptance_rate", 0)),
            ]

            for metric_name, metric_value in metric_names:
                record = MetricRecord(
                    metric_id=str(uuid.uuid4()),
                    episode_id=episode_id,
                    model_name=model_name,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    timestamp=datetime.utcnow(),
                )
                self._supabase_writer.write_metric(record)

        except Exception as e:
            logger.warning(f"Failed to write episode metrics: {e}")

    def write_episode_result(self, result: EpisodeResult) -> None:
        """Write a complete episode result to storage.

        Args:
            result: The episode result to write.
        """
        if not self._supabase_writer:
            return

        try:
            from salesbench.storage.supabase_writer import MetricRecord

            # Write comprehensive metrics
            metric_data = [
                ("final_score", result.final_score),
                ("total_turns", result.total_turns),
                ("total_accepts", result.total_accepts),
                ("total_rejects", result.total_rejects),
                ("total_calls", result.total_calls),
                ("dnc_violations", result.dnc_violations),
                ("duration_seconds", result.duration_seconds),
                ("acceptance_rate", result.acceptance_rate),
            ]

            for metric_name, metric_value in metric_data:
                record = MetricRecord(
                    metric_id=str(uuid.uuid4()),
                    episode_id=result.episode_id,
                    model_name=self.config.seller_model or "unknown",
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                    tags={
                        "benchmark_id": result.benchmark_id,
                        "episode_index": result.episode_index,
                        "status": result.status,
                    },
                    timestamp=datetime.utcnow(),
                )
                self._supabase_writer.write_metric(record)

        except Exception as e:
            logger.warning(f"Failed to write episode result: {e}")

    def write_benchmark_summary(
        self,
        benchmark_id: str,
        aggregate_metrics: dict,
        config: dict,
    ) -> None:
        """Write benchmark summary metrics to storage.

        Args:
            benchmark_id: Benchmark identifier.
            aggregate_metrics: Aggregate statistics.
            config: Benchmark configuration.
        """
        if not self._supabase_writer:
            return

        try:
            from salesbench.storage.supabase_writer import MetricRecord

            # Write aggregate metrics
            for metric_name, metric_value in aggregate_metrics.items():
                if isinstance(metric_value, (int, float)) and metric_value is not None:
                    record = MetricRecord(
                        metric_id=str(uuid.uuid4()),
                        episode_id=benchmark_id,  # Use benchmark_id as episode_id for aggregates
                        model_name=config.get("seller_model", "unknown"),
                        metric_name=f"aggregate_{metric_name}",
                        metric_value=float(metric_value),
                        tags={
                            "benchmark_id": benchmark_id,
                            "type": "aggregate",
                            "mode": config.get("mode", "unknown"),
                        },
                        timestamp=datetime.utcnow(),
                    )
                    self._supabase_writer.write_metric(record)

        except Exception as e:
            logger.warning(f"Failed to write benchmark summary: {e}")

    def get_grafana_trace_url(self, benchmark_id: str) -> Optional[str]:
        """Get Grafana trace URL for a benchmark.

        Args:
            benchmark_id: The benchmark ID to link to.

        Returns:
            URL string if Grafana is configured, None otherwise.
        """
        if not self._telemetry_manager:
            return None

        try:
            import os

            grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
            return f"{grafana_url}/explore?benchmark={benchmark_id}"
        except Exception:
            return None

    def __enter__(self) -> "IntegrationManager":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()
