"""Span definitions for SalesBench telemetry.

Provides structured spans for tracing episodes, calls, and tool executions.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Generator
import logging

from salesbench.telemetry.otel import get_tracer, get_meter

logger = logging.getLogger(__name__)


@dataclass
class SpanAttributes:
    """Common span attributes."""

    episode_id: Optional[str] = None
    seed: Optional[int] = None
    model_name: Optional[str] = None
    lead_id: Optional[str] = None
    call_id: Optional[str] = None
    tool_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in {
            "episode.id": self.episode_id,
            "episode.seed": self.seed,
            "model.name": self.model_name,
            "lead.id": self.lead_id,
            "call.id": self.call_id,
            "tool.name": self.tool_name,
        }.items() if v is not None}


class SpanManager:
    """Manages telemetry spans and metrics for SalesBench."""

    def __init__(self):
        self._tracer = None
        self._meter = None
        self._counters = {}
        self._histograms = {}

    @property
    def tracer(self):
        if self._tracer is None:
            self._tracer = get_tracer()
        return self._tracer

    @property
    def meter(self):
        if self._meter is None:
            self._meter = get_meter()
        return self._meter

    def _get_counter(self, name: str, description: str = ""):
        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(
                name,
                description=description,
            )
        return self._counters[name]

    def _get_histogram(self, name: str, description: str = "", unit: str = ""):
        if name not in self._histograms:
            self._histograms[name] = self.meter.create_histogram(
                name,
                description=description,
                unit=unit,
            )
        return self._histograms[name]

    @contextmanager
    def episode_span(
        self,
        episode_id: str,
        seed: int,
        model_name: str,
        num_leads: int = 100,
    ) -> Generator["EpisodeSpan", None, None]:
        """Create a span for an episode."""
        with self.tracer.start_as_current_span("episode") as span:
            episode = EpisodeSpan(
                span=span,
                manager=self,
                episode_id=episode_id,
                seed=seed,
                model_name=model_name,
            )

            span.set_attribute("episode.id", episode_id)
            span.set_attribute("episode.seed", seed)
            span.set_attribute("model.name", model_name)
            span.set_attribute("episode.num_leads", num_leads)

            # Increment episode counter
            self._get_counter(
                "salesbench.episodes.total",
                "Total number of episodes"
            ).add(1, {"model": model_name})

            try:
                yield episode
            finally:
                # Record episode duration
                if episode.duration_seconds:
                    self._get_histogram(
                        "salesbench.episode.duration",
                        "Episode duration in seconds",
                        "s"
                    ).record(episode.duration_seconds, {"model": model_name})

    @contextmanager
    def call_span(
        self,
        call_id: str,
        lead_id: str,
        episode_id: str,
    ) -> Generator["CallSpan", None, None]:
        """Create a span for a call."""
        with self.tracer.start_as_current_span("call") as span:
            call = CallSpan(
                span=span,
                manager=self,
                call_id=call_id,
                lead_id=lead_id,
            )

            span.set_attribute("call.id", call_id)
            span.set_attribute("lead.id", lead_id)
            span.set_attribute("episode.id", episode_id)

            # Increment call counter
            self._get_counter(
                "salesbench.calls.total",
                "Total number of calls"
            ).add(1)

            try:
                yield call
            finally:
                pass

    @contextmanager
    def tool_span(
        self,
        tool_name: str,
        episode_id: str,
        arguments: Optional[dict] = None,
    ) -> Generator["ToolSpan", None, None]:
        """Create a span for a tool execution."""
        with self.tracer.start_as_current_span(f"tool.{tool_name}") as span:
            tool = ToolSpan(
                span=span,
                manager=self,
                tool_name=tool_name,
            )

            span.set_attribute("tool.name", tool_name)
            span.set_attribute("episode.id", episode_id)

            # Increment tool counter
            self._get_counter(
                "salesbench.tools.total",
                "Total tool calls"
            ).add(1, {"tool": tool_name})

            try:
                yield tool
            except Exception as e:
                tool.record_error(str(e))
                raise
            finally:
                pass


@dataclass
class EpisodeSpan:
    """Span wrapper for an episode."""

    span: Any
    manager: SpanManager
    episode_id: str
    seed: int
    model_name: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def set_final_score(self, score: float) -> None:
        self.span.set_attribute("episode.final_score", score)
        self.manager._get_histogram(
            "salesbench.episode.score",
            "Episode final score"
        ).record(score, {"model": self.model_name})

    def set_metrics(self, metrics: dict) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.span.set_attribute(f"episode.metrics.{key}", value)

    def record_accept(self, plan_id: str, premium: float) -> None:
        self.span.add_event("plan_accepted", {"plan_id": plan_id, "premium": premium})
        self.manager._get_counter(
            "salesbench.accepts.total",
            "Total accepted plans"
        ).add(1, {"plan": plan_id, "model": self.model_name})

    def record_reject(self, plan_id: str) -> None:
        self.span.add_event("plan_rejected", {"plan_id": plan_id})
        self.manager._get_counter(
            "salesbench.rejects.total",
            "Total rejected plans"
        ).add(1, {"plan": plan_id, "model": self.model_name})

    def record_dnc_violation(self, lead_id: str) -> None:
        self.span.add_event("dnc_violation", {"lead_id": lead_id})
        self.manager._get_counter(
            "salesbench.dnc_violations.total",
            "Total DNC violations"
        ).add(1, {"model": self.model_name})

    def finish(self, final_score: float, metrics: dict) -> None:
        self.end_time = datetime.utcnow()
        self.set_final_score(final_score)
        self.set_metrics(metrics)


@dataclass
class CallSpan:
    """Span wrapper for a call."""

    span: Any
    manager: SpanManager
    call_id: str
    lead_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    offers_presented: int = 0

    def record_offer(self, plan_id: str, premium: float) -> None:
        self.offers_presented += 1
        self.span.add_event("offer_presented", {
            "plan_id": plan_id,
            "premium": premium,
            "offer_number": self.offers_presented,
        })

    def record_outcome(self, outcome: str, reason: Optional[str] = None) -> None:
        self.span.set_attribute("call.outcome", outcome)
        if reason:
            self.span.set_attribute("call.outcome_reason", reason)

        duration = (datetime.utcnow() - self.start_time).total_seconds()
        self.manager._get_histogram(
            "salesbench.call.duration",
            "Call duration in seconds",
            "s"
        ).record(duration, {"outcome": outcome})


@dataclass
class ToolSpan:
    """Span wrapper for a tool execution."""

    span: Any
    manager: SpanManager
    tool_name: str
    success: bool = True

    def set_result(self, success: bool, data: Optional[dict] = None) -> None:
        self.success = success
        self.span.set_attribute("tool.success", success)

        if success:
            self.manager._get_counter(
                "salesbench.tools.success",
                "Successful tool calls"
            ).add(1, {"tool": self.tool_name})
        else:
            self.manager._get_counter(
                "salesbench.tools.failure",
                "Failed tool calls"
            ).add(1, {"tool": self.tool_name})

    def record_error(self, error: str) -> None:
        self.success = False
        self.span.set_attribute("tool.success", False)
        self.span.set_attribute("tool.error", error)
        self.manager._get_counter(
            "salesbench.tools.failure",
            "Failed tool calls"
        ).add(1, {"tool": self.tool_name})


# Global span manager instance
_span_manager: Optional[SpanManager] = None


def get_span_manager() -> SpanManager:
    """Get or create the global span manager."""
    global _span_manager
    if _span_manager is None:
        _span_manager = SpanManager()
    return _span_manager
