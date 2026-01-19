"""Random events that can occur during calls.

Events add realism and unpredictability to buyer behavior,
making the benchmark more challenging and realistic.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable


class EventType(str, Enum):
    """Types of random events that can occur during calls."""

    # Interruptions
    SPOUSE_INVOLVEMENT = "spouse_involvement"
    CHILD_INTERRUPTION = "child_interruption"
    DOORBELL_RING = "doorbell_ring"
    PHONE_CALL = "phone_call"
    PET_DISTRACTION = "pet_distraction"

    # Technical issues
    BACKGROUND_NOISE = "background_noise"
    BAD_CONNECTION = "bad_connection"
    BATTERY_LOW = "battery_low"
    CALL_WAITING = "call_waiting"

    # Time constraints
    MEETING_SOON = "meeting_soon"
    APPOINTMENT_REMINDER = "appointment_reminder"
    DELIVERY_ARRIVED = "delivery_arrived"
    WORK_EMERGENCY = "work_emergency"

    # Emotional triggers
    POSITIVE_MEMORY = "positive_memory"
    NEGATIVE_MEMORY = "negative_memory"
    HEALTH_CONCERN = "health_concern"
    FAMILY_MENTION = "family_mention"

    # External influences
    SPOUSE_OPINION = "spouse_opinion"
    NEWS_REFERENCE = "news_reference"
    COMPETITOR_MENTION = "competitor_mention"


class EventImpact(str, Enum):
    """How the event impacts the call."""

    PATIENCE_DECREASE = "patience_decrease"
    PATIENCE_INCREASE = "patience_increase"
    INTEREST_DECREASE = "interest_decrease"
    INTEREST_INCREASE = "interest_increase"
    TRUST_DECREASE = "trust_decrease"
    TRUST_INCREASE = "trust_increase"
    CALL_END_RISK = "call_end_risk"
    DECISION_DELAY = "decision_delay"
    CLOSE_THRESHOLD_UP = "close_threshold_up"
    CLOSE_THRESHOLD_DOWN = "close_threshold_down"


@dataclass
class EventDefinition:
    """Definition of a random event."""

    event_type: EventType
    description: str
    probability: float  # Base probability per turn
    impacts: list[tuple[EventImpact, float]]  # (impact, magnitude)
    duration_turns: int = 1  # How long the effect lasts
    buyer_message: Optional[str] = None  # What buyer says
    prerequisite: Optional[Callable[["Persona"], bool]] = None  # Condition to trigger

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "description": self.description,
            "probability": self.probability,
            "impacts": [(i.value, m) for i, m in self.impacts],
            "duration_turns": self.duration_turns,
            "buyer_message": self.buyer_message,
        }


@dataclass
class ActiveEvent:
    """An event that is currently active during a call."""

    definition: EventDefinition
    triggered_at_turn: int
    remaining_turns: int
    applied_impacts: dict[EventImpact, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.definition.event_type.value,
            "triggered_at_turn": self.triggered_at_turn,
            "remaining_turns": self.remaining_turns,
            "applied_impacts": {k.value: v for k, v in self.applied_impacts.items()},
        }


# Event definitions with realistic probabilities
EVENT_DEFINITIONS: list[EventDefinition] = [
    # === Interruptions ===
    EventDefinition(
        event_type=EventType.SPOUSE_INVOLVEMENT,
        description="Spouse joins the call or asks what's going on",
        probability=0.08,
        impacts=[
            (EventImpact.DECISION_DELAY, 0.3),
            (EventImpact.INTEREST_INCREASE, 0.1),
        ],
        duration_turns=2,
        buyer_message="Hold on, my spouse wants to know who's calling...",
        prerequisite=lambda p: p.has_spouse,
    ),
    EventDefinition(
        event_type=EventType.CHILD_INTERRUPTION,
        description="Child interrupts the call needing attention",
        probability=0.10,
        impacts=[
            (EventImpact.PATIENCE_DECREASE, 0.2),
            (EventImpact.CALL_END_RISK, 0.15),
        ],
        duration_turns=1,
        buyer_message="Sorry, one moment - my kid needs something...",
        prerequisite=lambda p: p.num_dependents > 0,
    ),
    EventDefinition(
        event_type=EventType.DOORBELL_RING,
        description="Someone at the door",
        probability=0.05,
        impacts=[
            (EventImpact.PATIENCE_DECREASE, 0.15),
            (EventImpact.CALL_END_RISK, 0.1),
        ],
        duration_turns=1,
        buyer_message="There's someone at the door, can you hold on?",
    ),
    EventDefinition(
        event_type=EventType.PHONE_CALL,
        description="Another call comes in",
        probability=0.06,
        impacts=[
            (EventImpact.CALL_END_RISK, 0.2),
            (EventImpact.PATIENCE_DECREASE, 0.1),
        ],
        duration_turns=1,
        buyer_message="I'm getting another call, I might need to take it...",
    ),
    EventDefinition(
        event_type=EventType.PET_DISTRACTION,
        description="Pet causes a distraction",
        probability=0.04,
        impacts=[
            (EventImpact.PATIENCE_DECREASE, 0.1),
        ],
        duration_turns=1,
        buyer_message="Sorry, the dog is going crazy...",
    ),

    # === Technical Issues ===
    EventDefinition(
        event_type=EventType.BACKGROUND_NOISE,
        description="Noisy environment makes conversation difficult",
        probability=0.07,
        impacts=[
            (EventImpact.PATIENCE_DECREASE, 0.15),
            (EventImpact.TRUST_DECREASE, 0.05),
        ],
        duration_turns=2,
        buyer_message="It's really loud here, can you speak up?",
    ),
    EventDefinition(
        event_type=EventType.BAD_CONNECTION,
        description="Call connection becomes poor",
        probability=0.05,
        impacts=[
            (EventImpact.PATIENCE_DECREASE, 0.2),
            (EventImpact.CALL_END_RISK, 0.15),
        ],
        duration_turns=2,
        buyer_message="You're breaking up, can you hear me?",
    ),
    EventDefinition(
        event_type=EventType.BATTERY_LOW,
        description="Phone battery is running low",
        probability=0.03,
        impacts=[
            (EventImpact.CALL_END_RISK, 0.25),
            (EventImpact.PATIENCE_DECREASE, 0.1),
        ],
        duration_turns=3,
        buyer_message="My phone is about to die, you'll need to make this quick.",
    ),
    EventDefinition(
        event_type=EventType.CALL_WAITING,
        description="Call waiting beep interrupts",
        probability=0.04,
        impacts=[
            (EventImpact.CALL_END_RISK, 0.1),
        ],
        duration_turns=1,
        buyer_message="I have call waiting, what were you saying?",
    ),

    # === Time Constraints ===
    EventDefinition(
        event_type=EventType.MEETING_SOON,
        description="Buyer has a meeting coming up",
        probability=0.06,
        impacts=[
            (EventImpact.PATIENCE_DECREASE, 0.25),
            (EventImpact.CALL_END_RISK, 0.2),
        ],
        duration_turns=3,
        buyer_message="I have a meeting in a few minutes, so I can't talk long.",
    ),
    EventDefinition(
        event_type=EventType.APPOINTMENT_REMINDER,
        description="Calendar reminder pops up",
        probability=0.04,
        impacts=[
            (EventImpact.PATIENCE_DECREASE, 0.15),
            (EventImpact.CALL_END_RISK, 0.1),
        ],
        duration_turns=1,
        buyer_message="Oh, I just got a reminder - I need to leave soon.",
    ),
    EventDefinition(
        event_type=EventType.DELIVERY_ARRIVED,
        description="Package delivery arrives",
        probability=0.03,
        impacts=[
            (EventImpact.CALL_END_RISK, 0.15),
        ],
        duration_turns=1,
        buyer_message="Hold on, there's a delivery I need to sign for.",
    ),
    EventDefinition(
        event_type=EventType.WORK_EMERGENCY,
        description="Work emergency comes up",
        probability=0.04,
        impacts=[
            (EventImpact.CALL_END_RISK, 0.4),
            (EventImpact.PATIENCE_DECREASE, 0.3),
        ],
        duration_turns=1,
        buyer_message="Something just came up at work, I really need to go.",
    ),

    # === Emotional Triggers ===
    EventDefinition(
        event_type=EventType.POSITIVE_MEMORY,
        description="Discussion triggers a positive memory about insurance",
        probability=0.05,
        impacts=[
            (EventImpact.INTEREST_INCREASE, 0.2),
            (EventImpact.TRUST_INCREASE, 0.15),
        ],
        duration_turns=2,
        buyer_message="You know, my parents had good insurance and it really helped when...",
    ),
    EventDefinition(
        event_type=EventType.NEGATIVE_MEMORY,
        description="Discussion triggers a negative memory about insurance",
        probability=0.06,
        impacts=[
            (EventImpact.TRUST_DECREASE, 0.25),
            (EventImpact.INTEREST_DECREASE, 0.15),
        ],
        duration_turns=2,
        buyer_message="I had a bad experience with insurance before...",
    ),
    EventDefinition(
        event_type=EventType.HEALTH_CONCERN,
        description="Buyer mentions a health concern",
        probability=0.04,
        impacts=[
            (EventImpact.INTEREST_INCREASE, 0.25),
            (EventImpact.CLOSE_THRESHOLD_UP, 0.02),
        ],
        duration_turns=2,
        buyer_message="Actually, I've been thinking about this because of some health issues...",
    ),
    EventDefinition(
        event_type=EventType.FAMILY_MENTION,
        description="Buyer thinks about family protection",
        probability=0.07,
        impacts=[
            (EventImpact.INTEREST_INCREASE, 0.2),
            (EventImpact.PATIENCE_INCREASE, 0.1),
        ],
        duration_turns=2,
        buyer_message="When I think about my family's future...",
        prerequisite=lambda p: p.household_size > 1,
    ),

    # === External Influences ===
    EventDefinition(
        event_type=EventType.SPOUSE_OPINION,
        description="Spouse expresses an opinion about insurance",
        probability=0.05,
        impacts=[
            (EventImpact.DECISION_DELAY, 0.2),
            (EventImpact.INTEREST_DECREASE, 0.1),
        ],
        duration_turns=1,
        buyer_message="My spouse thinks we should shop around more...",
        prerequisite=lambda p: p.has_spouse,
    ),
    EventDefinition(
        event_type=EventType.NEWS_REFERENCE,
        description="Buyer mentions something from the news",
        probability=0.03,
        impacts=[
            (EventImpact.TRUST_DECREASE, 0.1),
        ],
        duration_turns=1,
        buyer_message="I saw something in the news about insurance companies...",
    ),
    EventDefinition(
        event_type=EventType.COMPETITOR_MENTION,
        description="Buyer mentions a competitor",
        probability=0.06,
        impacts=[
            (EventImpact.CLOSE_THRESHOLD_DOWN, 0.01),
            (EventImpact.TRUST_DECREASE, 0.05),
        ],
        duration_turns=1,
        buyer_message="I was also talking to another company...",
    ),
]


class RandomEventEngine:
    """Engine for generating and managing random events during calls."""

    def __init__(self, seed: int, event_multiplier: float = 1.0):
        """Initialize the event engine.

        Args:
            seed: Random seed for reproducibility.
            event_multiplier: Multiplier for event probabilities (1.0 = normal).
        """
        self._rng = random.Random(seed)
        self.event_multiplier = event_multiplier
        self._event_definitions = {e.event_type: e for e in EVENT_DEFINITIONS}

    def check_for_events(
        self,
        persona: "Persona",
        current_turn: int,
        active_events: list[ActiveEvent],
    ) -> list[ActiveEvent]:
        """Check if any new events trigger.

        Args:
            persona: The buyer persona.
            current_turn: Current turn number in the call.
            active_events: Currently active events.

        Returns:
            List of newly triggered events.
        """
        new_events = []

        # Don't trigger too many events at once
        active_types = {e.definition.event_type for e in active_events}

        for definition in EVENT_DEFINITIONS:
            # Skip if already active
            if definition.event_type in active_types:
                continue

            # Check prerequisite
            if definition.prerequisite and not definition.prerequisite(persona):
                continue

            # Check probability
            adjusted_prob = definition.probability * self.event_multiplier
            if self._rng.random() < adjusted_prob:
                event = ActiveEvent(
                    definition=definition,
                    triggered_at_turn=current_turn,
                    remaining_turns=definition.duration_turns,
                )
                new_events.append(event)

                # Limit to 1-2 new events per check
                if len(new_events) >= 2:
                    break

        return new_events

    def apply_event_impacts(
        self,
        event: ActiveEvent,
        persona: "Persona",
    ) -> dict[str, float]:
        """Apply event impacts to persona's hidden state.

        Args:
            event: The event to apply.
            persona: The persona to modify.

        Returns:
            Dict of applied changes.
        """
        changes = {}

        for impact, magnitude in event.definition.impacts:
            # Add some randomness to magnitude
            actual_magnitude = magnitude * self._rng.uniform(0.8, 1.2)

            if impact == EventImpact.PATIENCE_DECREASE:
                old = persona.hidden.patience
                persona.hidden.patience = max(0.0, old - actual_magnitude)
                changes["patience"] = persona.hidden.patience - old

            elif impact == EventImpact.PATIENCE_INCREASE:
                old = persona.hidden.patience
                persona.hidden.patience = min(1.0, old + actual_magnitude)
                changes["patience"] = persona.hidden.patience - old

            elif impact == EventImpact.INTEREST_DECREASE:
                old = persona.hidden.interest
                persona.hidden.interest = max(0.0, old - actual_magnitude)
                changes["interest"] = persona.hidden.interest - old

            elif impact == EventImpact.INTEREST_INCREASE:
                old = persona.hidden.interest
                persona.hidden.interest = min(1.0, old + actual_magnitude)
                changes["interest"] = persona.hidden.interest - old

            elif impact == EventImpact.TRUST_DECREASE:
                old = persona.hidden.trust
                persona.hidden.trust = max(0.0, old - actual_magnitude)
                changes["trust"] = persona.hidden.trust - old

            elif impact == EventImpact.TRUST_INCREASE:
                old = persona.hidden.trust
                persona.hidden.trust = min(1.0, old + actual_magnitude)
                changes["trust"] = persona.hidden.trust - old

            elif impact == EventImpact.CLOSE_THRESHOLD_UP:
                old = persona.hidden.close_threshold
                persona.hidden.close_threshold = min(0.2, old + actual_magnitude)
                changes["close_threshold"] = persona.hidden.close_threshold - old

            elif impact == EventImpact.CLOSE_THRESHOLD_DOWN:
                old = persona.hidden.close_threshold
                persona.hidden.close_threshold = max(0.01, old - actual_magnitude)
                changes["close_threshold"] = persona.hidden.close_threshold - old

            elif impact in (EventImpact.CALL_END_RISK, EventImpact.DECISION_DELAY):
                # These are checked separately
                changes[impact.value] = actual_magnitude

            event.applied_impacts[impact] = actual_magnitude

        return changes

    def should_end_call(self, active_events: list[ActiveEvent]) -> tuple[bool, Optional[str]]:
        """Check if any active event should end the call.

        Args:
            active_events: Currently active events.

        Returns:
            Tuple of (should_end, reason).
        """
        for event in active_events:
            for impact, magnitude in event.applied_impacts.items():
                if impact == EventImpact.CALL_END_RISK:
                    if self._rng.random() < magnitude:
                        return True, f"Call ended due to: {event.definition.description}"
        return False, None

    def tick_events(self, active_events: list[ActiveEvent]) -> list[ActiveEvent]:
        """Advance time for active events.

        Args:
            active_events: Currently active events.

        Returns:
            Events still active after tick.
        """
        still_active = []
        for event in active_events:
            event.remaining_turns -= 1
            if event.remaining_turns > 0:
                still_active.append(event)
        return still_active

    def get_buyer_messages(self, events: list[ActiveEvent]) -> list[str]:
        """Get buyer messages for triggered events.

        Args:
            events: Newly triggered events.

        Returns:
            List of buyer messages to display.
        """
        messages = []
        for event in events:
            if event.definition.buyer_message:
                messages.append(event.definition.buyer_message)
        return messages


@dataclass
class CallEventLog:
    """Log of all events during a call."""

    call_id: str
    events: list[dict] = field(default_factory=list)

    def add_event(self, event: ActiveEvent, turn: int, impacts: dict) -> None:
        """Add an event to the log."""
        self.events.append({
            "turn": turn,
            "event_type": event.definition.event_type.value,
            "description": event.definition.description,
            "buyer_message": event.definition.buyer_message,
            "impacts": impacts,
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "call_id": self.call_id,
            "event_count": len(self.events),
            "events": self.events,
        }


# Type hint import at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from salesbench.envs.sales_mvp.personas import Persona
