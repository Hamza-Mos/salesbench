"""Event logging system for SalesBench metrics.

Provides comprehensive event tracking for:
- Episode lifecycle events
- Call events (start, end, offers, decisions)
- Tool usage events
- Random events
- Error events
"""

from salesbench.events.aggregators import (
    CallAggregator,
    EpisodeAggregator,
    EventAggregator,
    PerformanceAggregator,
)
from salesbench.events.event_log import (
    Event,
    EventEmitter,
    EventLog,
    EventSubscriber,
    EventType,
)
from salesbench.events.event_types import (
    CallEvent,
    DecisionEvent,
    EpisodeEvent,
    ErrorEvent,
    MetricEvent,
    OfferEvent,
    RandomEvent,
    ToolEvent,
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
