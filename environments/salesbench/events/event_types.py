"""Specific event type definitions."""

from dataclasses import dataclass, field
from typing import Any, Optional

from salesbench.events.event_log import Event, EventType, EventLog


@dataclass
class EpisodeEvent:
    """Episode lifecycle events."""

    episode_id: str
    seed: int
    agent_type: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        log: EventLog,
        episode_id: str,
        seed: int,
        agent_type: str,
        config: Optional[dict] = None,
    ) -> Event:
        """Log episode start."""
        return log.log(
            event_type=EventType.EPISODE_START,
            data={
                "seed": seed,
                "agent_type": agent_type,
                "config": config or {},
            },
            episode_id=episode_id,
            tags=["lifecycle"],
        )

    @classmethod
    def end(
        cls,
        log: EventLog,
        episode_id: str,
        outcome: str,
        total_reward: float,
        metrics: dict[str, Any],
    ) -> Event:
        """Log episode end."""
        return log.log(
            event_type=EventType.EPISODE_END,
            data={
                "outcome": outcome,
                "total_reward": total_reward,
                "metrics": metrics,
            },
            episode_id=episode_id,
            tags=["lifecycle"],
        )

    @classmethod
    def day_start(
        cls,
        log: EventLog,
        episode_id: str,
        day: int,
        remaining_budget: dict[str, int],
    ) -> Event:
        """Log day start."""
        return log.log(
            event_type=EventType.DAY_START,
            data={
                "day": day,
                "remaining_budget": remaining_budget,
            },
            episode_id=episode_id,
            tags=["lifecycle", "day"],
        )

    @classmethod
    def day_end(
        cls,
        log: EventLog,
        episode_id: str,
        day: int,
        summary: dict[str, Any],
    ) -> Event:
        """Log day end."""
        return log.log(
            event_type=EventType.DAY_END,
            data={
                "day": day,
                "summary": summary,
            },
            episode_id=episode_id,
            tags=["lifecycle", "day"],
        )


@dataclass
class CallEvent:
    """Call lifecycle events."""

    @classmethod
    def start(
        cls,
        log: EventLog,
        episode_id: str,
        call_id: str,
        lead_id: str,
        lead_info: dict[str, Any],
        turn: int,
    ) -> Event:
        """Log call start."""
        return log.log(
            event_type=EventType.CALL_START,
            data={
                "lead_info": lead_info,
            },
            episode_id=episode_id,
            call_id=call_id,
            lead_id=lead_id,
            turn=turn,
            tags=["call"],
        )

    @classmethod
    def end(
        cls,
        log: EventLog,
        episode_id: str,
        call_id: str,
        lead_id: str,
        outcome: str,
        duration_minutes: int,
        offers_count: int,
        turn: int,
    ) -> Event:
        """Log call end."""
        return log.log(
            event_type=EventType.CALL_END,
            data={
                "outcome": outcome,
                "duration_minutes": duration_minutes,
                "offers_count": offers_count,
            },
            episode_id=episode_id,
            call_id=call_id,
            lead_id=lead_id,
            turn=turn,
            tags=["call"],
        )


@dataclass
class ToolEvent:
    """Tool usage events."""

    @classmethod
    def call(
        cls,
        log: EventLog,
        episode_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        call_id: Optional[str] = None,
        turn: int = 0,
    ) -> Event:
        """Log tool call."""
        return log.log(
            event_type=EventType.TOOL_CALL,
            data={
                "tool_name": tool_name,
                "arguments": arguments,
            },
            episode_id=episode_id,
            call_id=call_id,
            turn=turn,
            tags=["tool", tool_name.split(".")[0]],
        )

    @classmethod
    def result(
        cls,
        log: EventLog,
        episode_id: str,
        tool_name: str,
        success: bool,
        result: Any,
        call_id: Optional[str] = None,
        turn: int = 0,
    ) -> Event:
        """Log tool result."""
        return log.log(
            event_type=EventType.TOOL_RESULT,
            data={
                "tool_name": tool_name,
                "success": success,
                "result": result,
            },
            episode_id=episode_id,
            call_id=call_id,
            turn=turn,
            tags=["tool", tool_name.split(".")[0]],
        )

    @classmethod
    def error(
        cls,
        log: EventLog,
        episode_id: str,
        tool_name: str,
        error: str,
        call_id: Optional[str] = None,
        turn: int = 0,
    ) -> Event:
        """Log tool error."""
        return log.log(
            event_type=EventType.TOOL_ERROR,
            data={
                "tool_name": tool_name,
                "error": error,
            },
            episode_id=episode_id,
            call_id=call_id,
            turn=turn,
            tags=["tool", "error"],
        )


@dataclass
class OfferEvent:
    """Offer presentation events."""

    @classmethod
    def presented(
        cls,
        log: EventLog,
        episode_id: str,
        call_id: str,
        lead_id: str,
        offer: dict[str, Any],
        turn: int,
    ) -> Event:
        """Log offer presented."""
        return log.log(
            event_type=EventType.OFFER_PRESENTED,
            data={
                "offer": offer,
            },
            episode_id=episode_id,
            call_id=call_id,
            lead_id=lead_id,
            turn=turn,
            tags=["offer", offer.get("plan_id", "unknown")],
        )


@dataclass
class DecisionEvent:
    """Buyer decision events."""

    @classmethod
    def made(
        cls,
        log: EventLog,
        episode_id: str,
        call_id: str,
        lead_id: str,
        decision: str,
        reason: Optional[str],
        offer: dict[str, Any],
        turn: int,
    ) -> Event:
        """Log buyer decision."""
        return log.log(
            event_type=EventType.BUYER_DECISION,
            data={
                "decision": decision,
                "reason": reason,
                "offer": offer,
            },
            episode_id=episode_id,
            call_id=call_id,
            lead_id=lead_id,
            turn=turn,
            tags=["decision", decision],
        )


@dataclass
class RandomEvent:
    """Random events during calls."""

    @classmethod
    def triggered(
        cls,
        log: EventLog,
        episode_id: str,
        call_id: str,
        lead_id: str,
        event_type: str,
        description: str,
        impacts: dict[str, float],
        buyer_message: Optional[str],
        turn: int,
    ) -> Event:
        """Log random event triggered."""
        return log.log(
            event_type=EventType.RANDOM_EVENT,
            data={
                "random_event_type": event_type,
                "description": description,
                "impacts": impacts,
                "buyer_message": buyer_message,
            },
            episode_id=episode_id,
            call_id=call_id,
            lead_id=lead_id,
            turn=turn,
            tags=["random_event", event_type],
        )


@dataclass
class ErrorEvent:
    """Error and warning events."""

    @classmethod
    def error(
        cls,
        log: EventLog,
        episode_id: str,
        error_type: str,
        message: str,
        details: Optional[dict] = None,
        call_id: Optional[str] = None,
    ) -> Event:
        """Log an error."""
        return log.log(
            event_type=EventType.ERROR,
            data={
                "error_type": error_type,
                "message": message,
                "details": details or {},
            },
            episode_id=episode_id,
            call_id=call_id,
            tags=["error", error_type],
        )

    @classmethod
    def warning(
        cls,
        log: EventLog,
        episode_id: str,
        warning_type: str,
        message: str,
        details: Optional[dict] = None,
        call_id: Optional[str] = None,
    ) -> Event:
        """Log a warning."""
        return log.log(
            event_type=EventType.WARNING,
            data={
                "warning_type": warning_type,
                "message": message,
                "details": details or {},
            },
            episode_id=episode_id,
            call_id=call_id,
            tags=["warning", warning_type],
        )


@dataclass
class MetricEvent:
    """Metric recording events."""

    @classmethod
    def record(
        cls,
        log: EventLog,
        episode_id: str,
        metric_name: str,
        value: float,
        unit: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> Event:
        """Log a metric."""
        return log.log(
            event_type=EventType.METRIC,
            data={
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "labels": labels or {},
            },
            episode_id=episode_id,
            tags=["metric", metric_name],
        )

    @classmethod
    def checkpoint(
        cls,
        log: EventLog,
        episode_id: str,
        checkpoint_name: str,
        state: dict[str, Any],
    ) -> Event:
        """Log a checkpoint."""
        return log.log(
            event_type=EventType.CHECKPOINT,
            data={
                "checkpoint_name": checkpoint_name,
                "state": state,
            },
            episode_id=episode_id,
            tags=["checkpoint", checkpoint_name],
        )
