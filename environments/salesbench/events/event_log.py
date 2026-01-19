"""Core event logging infrastructure."""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict


class EventType(str, Enum):
    """High-level event categories."""

    # Episode lifecycle
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    DAY_START = "day_start"
    DAY_END = "day_end"

    # Call events
    CALL_START = "call_start"
    CALL_END = "call_end"
    OFFER_PRESENTED = "offer_presented"
    BUYER_DECISION = "buyer_decision"

    # Tool events
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # Random events
    RANDOM_EVENT = "random_event"

    # Agent events
    AGENT_ACTION = "agent_action"
    AGENT_THINKING = "agent_thinking"

    # System events
    ERROR = "error"
    WARNING = "warning"
    METRIC = "metric"
    CHECKPOINT = "checkpoint"


@dataclass
class Event:
    """A single logged event."""

    event_id: str
    event_type: EventType
    timestamp: float  # Unix timestamp
    data: dict[str, Any]

    # Optional context
    episode_id: Optional[str] = None
    call_id: Optional[str] = None
    lead_id: Optional[str] = None
    turn: Optional[int] = None

    # Metadata
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "episode_id": self.episode_id,
            "call_id": self.call_id,
            "lead_id": self.lead_id,
            "turn": self.turn,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            data=data["data"],
            episode_id=data.get("episode_id"),
            call_id=data.get("call_id"),
            lead_id=data.get("lead_id"),
            turn=data.get("turn"),
            tags=data.get("tags", []),
        )


class EventSubscriber(ABC):
    """Abstract base class for event subscribers."""

    @abstractmethod
    def on_event(self, event: Event) -> None:
        """Handle an event.

        Args:
            event: The event to handle.
        """
        pass


class EventEmitter:
    """Emits events to subscribers."""

    def __init__(self):
        """Initialize the emitter."""
        self._subscribers: list[EventSubscriber] = []
        self._type_subscribers: dict[EventType, list[EventSubscriber]] = defaultdict(list)
        self._filters: list[Callable[[Event], bool]] = []

    def subscribe(
        self,
        subscriber: EventSubscriber,
        event_types: Optional[list[EventType]] = None,
    ) -> None:
        """Subscribe to events.

        Args:
            subscriber: The subscriber.
            event_types: Optional list of event types to subscribe to.
                        If None, subscribes to all events.
        """
        if event_types is None:
            self._subscribers.append(subscriber)
        else:
            for event_type in event_types:
                self._type_subscribers[event_type].append(subscriber)

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        """Unsubscribe from events."""
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
        for subs in self._type_subscribers.values():
            if subscriber in subs:
                subs.remove(subscriber)

    def add_filter(self, filter_fn: Callable[[Event], bool]) -> None:
        """Add a filter for events.

        Args:
            filter_fn: Function that returns True if event should be emitted.
        """
        self._filters.append(filter_fn)

    def emit(self, event: Event) -> None:
        """Emit an event to subscribers.

        Args:
            event: The event to emit.
        """
        # Check filters
        for filter_fn in self._filters:
            if not filter_fn(event):
                return

        # Notify global subscribers
        for subscriber in self._subscribers:
            subscriber.on_event(event)

        # Notify type-specific subscribers
        for subscriber in self._type_subscribers.get(event.event_type, []):
            subscriber.on_event(event)


class EventLog:
    """Persistent event log with query capabilities."""

    def __init__(
        self,
        max_events: int = 10000,
        auto_flush_count: int = 100,
    ):
        """Initialize the event log.

        Args:
            max_events: Maximum events to keep in memory.
            auto_flush_count: Auto-flush after this many events.
        """
        self._events: list[Event] = []
        self._max_events = max_events
        self._auto_flush_count = auto_flush_count
        self._flush_callbacks: list[Callable[[list[Event]], None]] = []
        self._emitter = EventEmitter()

        # Indexes for fast queries
        self._by_episode: dict[str, list[Event]] = defaultdict(list)
        self._by_call: dict[str, list[Event]] = defaultdict(list)
        self._by_type: dict[EventType, list[Event]] = defaultdict(list)

    @property
    def emitter(self) -> EventEmitter:
        """Get the event emitter for subscriptions."""
        return self._emitter

    def log(
        self,
        event_type: EventType,
        data: dict[str, Any],
        episode_id: Optional[str] = None,
        call_id: Optional[str] = None,
        lead_id: Optional[str] = None,
        turn: Optional[int] = None,
        tags: Optional[list[str]] = None,
    ) -> Event:
        """Log an event.

        Args:
            event_type: Type of event.
            data: Event data.
            episode_id: Optional episode ID.
            call_id: Optional call ID.
            lead_id: Optional lead ID.
            turn: Optional turn number.
            tags: Optional tags.

        Returns:
            The created event.
        """
        event = Event(
            event_id=uuid.uuid4().hex[:12],
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            episode_id=episode_id,
            call_id=call_id,
            lead_id=lead_id,
            turn=turn,
            tags=tags or [],
        )

        self._events.append(event)

        # Update indexes
        if episode_id:
            self._by_episode[episode_id].append(event)
        if call_id:
            self._by_call[call_id].append(event)
        self._by_type[event_type].append(event)

        # Emit to subscribers
        self._emitter.emit(event)

        # Auto-flush if needed
        if len(self._events) >= self._auto_flush_count:
            self._auto_flush()

        # Trim if too many events
        if len(self._events) > self._max_events:
            self._trim_events()

        return event

    def query(
        self,
        event_type: Optional[EventType] = None,
        episode_id: Optional[str] = None,
        call_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        tags: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[Event]:
        """Query events with filters.

        Args:
            event_type: Filter by event type.
            episode_id: Filter by episode ID.
            call_id: Filter by call ID.
            start_time: Filter events after this time.
            end_time: Filter events before this time.
            tags: Filter by tags (any match).
            limit: Maximum events to return.

        Returns:
            Matching events.
        """
        # Start with appropriate index
        if call_id:
            candidates = self._by_call.get(call_id, [])
        elif episode_id:
            candidates = self._by_episode.get(episode_id, [])
        elif event_type:
            candidates = self._by_type.get(event_type, [])
        else:
            candidates = self._events

        results = []
        for event in candidates:
            # Apply filters
            if event_type and event.event_type != event_type:
                continue
            if episode_id and event.episode_id != episode_id:
                continue
            if call_id and event.call_id != call_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if tags and not any(t in event.tags for t in tags):
                continue

            results.append(event)
            if len(results) >= limit:
                break

        return results

    def get_episode_events(self, episode_id: str) -> list[Event]:
        """Get all events for an episode."""
        return self._by_episode.get(episode_id, [])

    def get_call_events(self, call_id: str) -> list[Event]:
        """Get all events for a call."""
        return self._by_call.get(call_id, [])

    def get_events_by_type(self, event_type: EventType) -> list[Event]:
        """Get all events of a type."""
        return self._by_type.get(event_type, [])

    def add_flush_callback(self, callback: Callable[[list[Event]], None]) -> None:
        """Add a callback to be called on flush.

        Args:
            callback: Function that receives events to flush.
        """
        self._flush_callbacks.append(callback)

    def flush(self) -> list[Event]:
        """Flush all events and return them.

        Returns:
            List of flushed events.
        """
        events = self._events.copy()

        for callback in self._flush_callbacks:
            callback(events)

        return events

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()
        self._by_episode.clear()
        self._by_call.clear()
        self._by_type.clear()

    def _auto_flush(self) -> None:
        """Auto-flush older events."""
        if len(self._events) < self._auto_flush_count:
            return

        # Get events to flush (older half)
        flush_count = len(self._events) // 2
        to_flush = self._events[:flush_count]

        for callback in self._flush_callbacks:
            callback(to_flush)

    def _trim_events(self) -> None:
        """Trim events when over max capacity."""
        if len(self._events) <= self._max_events:
            return

        # Remove oldest events
        trim_count = len(self._events) - self._max_events
        self._events = self._events[trim_count:]

        # Rebuild indexes (expensive but rare)
        self._by_episode.clear()
        self._by_call.clear()
        self._by_type.clear()

        for event in self._events:
            if event.episode_id:
                self._by_episode[event.episode_id].append(event)
            if event.call_id:
                self._by_call[event.call_id].append(event)
            self._by_type[event.event_type].append(event)

    def to_list(self) -> list[dict[str, Any]]:
        """Convert all events to list of dicts."""
        return [e.to_dict() for e in self._events]

    def __len__(self) -> int:
        """Return number of events."""
        return len(self._events)
