"""Event logging system for SalesBench metrics.

Provides comprehensive event tracking for:
- Episode lifecycle events
- Call events (start, end, offers, decisions)
- Tool usage events
- Random events
- Error events
"""

from salesbench.events.event_log import (
    Event,
    EventType,
    EventLog,
    EventEmitter,
    EventSubscriber,
)
from salesbench.events.event_types import (
    EpisodeEvent,
    CallEvent,
    ToolEvent,
    OfferEvent,
    DecisionEvent,
    RandomEvent,
    ErrorEvent,
    MetricEvent,
)
from salesbench.events.aggregators import (
    EventAggregator,
    EpisodeAggregator,
    CallAggregator,
    PerformanceAggregator,
)

__all__ = [
    # Core
    "Event",
    "EventType",
    "EventLog",
    "EventEmitter",
    "EventSubscriber",
    # Event types
    "EpisodeEvent",
    "CallEvent",
    "ToolEvent",
    "OfferEvent",
    "DecisionEvent",
    "RandomEvent",
    "ErrorEvent",
    "MetricEvent",
    # Aggregators
    "EventAggregator",
    "EpisodeAggregator",
    "CallAggregator",
    "PerformanceAggregator",
]
