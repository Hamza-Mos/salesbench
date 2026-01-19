"""Supabase storage integration for SalesBench.

Provides batch writing of episodes, events, and metrics to Supabase.
This is an optional outer-layer integration kept separate from the
publishable environment package.
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, List
from queue import Queue
from threading import Thread, Event
import logging

logger = logging.getLogger(__name__)


@dataclass
class SupabaseConfig:
    """Configuration for Supabase connection."""

    url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    anon_key: str = field(default_factory=lambda: os.getenv("SUPABASE_ANON_KEY", ""))
    service_role_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))

    # Table names
    episodes_table: str = "salesbench_episodes"
    events_table: str = "salesbench_events"
    metrics_table: str = "salesbench_metrics"
    leads_table: str = "salesbench_leads"
    calls_table: str = "salesbench_calls"

    # Batch settings
    batch_size: int = 100
    flush_interval_seconds: float = 5.0

    def validate(self) -> bool:
        """Check if configuration is valid."""
        return bool(self.url and (self.anon_key or self.service_role_key))


@dataclass
class EpisodeRecord:
    """Record for an episode."""

    episode_id: str
    seed: int
    model_name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    num_leads: int = 100
    total_days: int = 10
    final_score: Optional[float] = None
    metrics: Optional[dict] = None
    config: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "model_name": self.model_name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "num_leads": self.num_leads,
            "total_days": self.total_days,
            "final_score": self.final_score,
            "metrics": json.dumps(self.metrics) if self.metrics else None,
            "config": json.dumps(self.config) if self.config else None,
        }


@dataclass
class EventRecord:
    """Record for an event."""

    event_id: str
    episode_id: str
    event_type: str
    timestamp: datetime
    data: dict

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "episode_id": self.episode_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": json.dumps(self.data),
        }


@dataclass
class MetricRecord:
    """Record for a metric."""

    metric_id: str
    episode_id: str
    model_name: str
    metric_name: str
    metric_value: float
    tags: Optional[dict] = None
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "metric_id": self.metric_id,
            "episode_id": self.episode_id,
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "tags": json.dumps(self.tags) if self.tags else None,
            "timestamp": (self.timestamp or datetime.utcnow()).isoformat(),
        }


class BatchWriter:
    """Batched async writer for database records."""

    def __init__(self, config: SupabaseConfig):
        self.config = config
        self._queue: Queue = Queue()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._client = None

    def _get_client(self):
        """Get or create Supabase client."""
        if self._client is None:
            try:
                from supabase import create_client
                key = self.config.service_role_key or self.config.anon_key
                self._client = create_client(self.config.url, key)
            except ImportError:
                raise ImportError("supabase package required. Install with: pip install supabase")
        return self._client

    def start(self) -> None:
        """Start the background writer thread."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        logger.info("BatchWriter started")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the background writer and flush remaining records."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("BatchWriter stopped")

    def _writer_loop(self) -> None:
        """Background loop that flushes batches."""
        batch: List[tuple] = []

        while not self._stop_event.is_set():
            try:
                # Collect records up to batch size
                while len(batch) < self.config.batch_size:
                    try:
                        record = self._queue.get(timeout=self.config.flush_interval_seconds)
                        batch.append(record)
                    except:
                        break

                if batch:
                    self._flush_batch(batch)
                    batch = []

            except Exception as e:
                logger.error(f"BatchWriter error: {e}")

        # Final flush
        while not self._queue.empty():
            batch.append(self._queue.get_nowait())
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[tuple]) -> None:
        """Flush a batch of records to Supabase."""
        if not batch:
            return

        # Group by table
        by_table: dict[str, list] = {}
        for table, record in batch:
            if table not in by_table:
                by_table[table] = []
            by_table[table].append(record)

        client = self._get_client()

        for table, records in by_table.items():
            try:
                client.table(table).insert(records).execute()
                logger.debug(f"Flushed {len(records)} records to {table}")
            except Exception as e:
                logger.error(f"Failed to flush to {table}: {e}")

    def write(self, table: str, record: dict) -> None:
        """Queue a record for writing."""
        self._queue.put((table, record))


class SupabaseWriter:
    """High-level writer for SalesBench data."""

    def __init__(self, config: Optional[SupabaseConfig] = None):
        self.config = config or SupabaseConfig()
        self._batch_writer: Optional[BatchWriter] = None
        self._enabled = self.config.validate()

    @property
    def enabled(self) -> bool:
        """Check if Supabase writing is enabled."""
        return self._enabled

    def start(self) -> None:
        """Start the writer."""
        if not self._enabled:
            logger.warning("Supabase not configured, writer disabled")
            return

        self._batch_writer = BatchWriter(self.config)
        self._batch_writer.start()

    def stop(self) -> None:
        """Stop the writer."""
        if self._batch_writer:
            self._batch_writer.stop()
            self._batch_writer = None

    def write_episode(self, record: EpisodeRecord) -> None:
        """Write an episode record."""
        if self._batch_writer:
            self._batch_writer.write(self.config.episodes_table, record.to_dict())

    def write_event(self, record: EventRecord) -> None:
        """Write an event record."""
        if self._batch_writer:
            self._batch_writer.write(self.config.events_table, record.to_dict())

    def write_metric(self, record: MetricRecord) -> None:
        """Write a metric record."""
        if self._batch_writer:
            self._batch_writer.write(self.config.metrics_table, record.to_dict())

    def write_events_batch(self, records: List[EventRecord]) -> None:
        """Write multiple events."""
        for record in records:
            self.write_event(record)

    def write_metrics_batch(self, records: List[MetricRecord]) -> None:
        """Write multiple metrics."""
        for record in records:
            self.write_metric(record)

    def __enter__(self) -> "SupabaseWriter":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
