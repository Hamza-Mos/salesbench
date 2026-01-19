"""Core type definitions for SalesBench."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NewType, Optional
import uuid


# ID types
LeadID = NewType("LeadID", str)
CallID = NewType("CallID", str)
AppointmentID = NewType("AppointmentID", str)


def generate_lead_id() -> LeadID:
    """Generate a unique lead ID."""
    return LeadID(f"lead_{uuid.uuid4().hex[:8]}")


def generate_call_id() -> CallID:
    """Generate a unique call ID."""
    return CallID(f"call_{uuid.uuid4().hex[:8]}")


def generate_appointment_id() -> AppointmentID:
    """Generate a unique appointment ID."""
    return AppointmentID(f"appt_{uuid.uuid4().hex[:8]}")


class BuyerDecision(str, Enum):
    """Buyer decision enum - the only outputs buyer can provide."""

    ACCEPT_PLAN = "accept_plan"
    REJECT_PLAN = "reject_plan"
    END_CALL = "end_call"


class NextStep(str, Enum):
    """Next step options for plan offers."""

    SCHEDULE_FOLLOWUP = "schedule_followup"
    REQUEST_INFO = "request_info"
    CLOSE_NOW = "close_now"


class PlanType(str, Enum):
    """Insurance plan types."""

    TERM = "TERM"  # Term life insurance
    WHOLE = "WHOLE"  # Whole life insurance
    UL = "UL"  # Universal life
    VUL = "VUL"  # Variable universal life
    LTC = "LTC"  # Long-term care
    DI = "DI"  # Disability insurance


class LeadTemperature(str, Enum):
    """Lead temperature classification."""

    HOT = "hot"  # 3% - Ready to buy
    WARM = "warm"  # 12% - Interested
    LUKEWARM = "lukewarm"  # 35% - Open to discussion
    COLD = "cold"  # 40% - Skeptical
    HOSTILE = "hostile"  # 10% - Actively resistant


class ObjectionStyle(str, Enum):
    """How the buyer expresses objections."""

    DIRECT = "direct"  # "No, I'm not interested"
    INDIRECT = "indirect"  # "Let me think about it"
    QUESTIONING = "questioning"  # "Why would I need this?"
    PRICE_FOCUSED = "price_focused"  # "That's too expensive"
    TRUST_ISSUES = "trust_issues"  # "I don't trust insurance companies"


class RiskClass(str, Enum):
    """Insurance risk classification."""

    PREFERRED_PLUS = "preferred_plus"  # Best rates
    PREFERRED = "preferred"  # Good rates
    STANDARD_PLUS = "standard_plus"  # Average rates
    STANDARD = "standard"  # Higher rates
    SUBSTANDARD = "substandard"  # Highest rates


class CoverageTier(str, Enum):
    """Coverage amount tiers."""

    BASIC = "basic"  # $100K
    STANDARD = "standard"  # $250K
    ENHANCED = "enhanced"  # $500K
    PREMIUM = "premium"  # $1M
    ELITE = "elite"  # $2M+


@dataclass
class ToolCall:
    """A tool call from the seller agent."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            arguments=data["arguments"],
            call_id=data.get("call_id", uuid.uuid4().hex[:8]),
        )


@dataclass
class ToolResult:
    """Result from executing a tool."""

    call_id: str
    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "call_id": self.call_id,
            "success": self.success,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        """Create from dictionary."""
        return cls(
            call_id=data["call_id"],
            success=data["success"],
            data=data.get("data"),
            error=data.get("error"),
        )


@dataclass
class PlanOffer:
    """A structured insurance plan offer."""

    plan_id: PlanType
    monthly_premium: float
    next_step: NextStep
    coverage_amount: float
    term_years: Optional[int] = None  # For TERM plans
    cash_value_year_10: Optional[float] = None  # For WHOLE/UL/VUL
    benefit_period: Optional[str] = None  # For DI/LTC
    waiting_period_days: Optional[int] = None  # For DI

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "plan_id": self.plan_id.value,
            "monthly_premium": self.monthly_premium,
            "next_step": self.next_step.value,
            "coverage_amount": self.coverage_amount,
        }
        if self.term_years is not None:
            result["term_years"] = self.term_years
        if self.cash_value_year_10 is not None:
            result["cash_value_year_10"] = self.cash_value_year_10
        if self.benefit_period is not None:
            result["benefit_period"] = self.benefit_period
        if self.waiting_period_days is not None:
            result["waiting_period_days"] = self.waiting_period_days
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanOffer":
        """Create from dictionary."""
        return cls(
            plan_id=PlanType(data["plan_id"]),
            monthly_premium=data["monthly_premium"],
            next_step=NextStep(data["next_step"]),
            coverage_amount=data["coverage_amount"],
            term_years=data.get("term_years"),
            cash_value_year_10=data.get("cash_value_year_10"),
            benefit_period=data.get("benefit_period"),
            waiting_period_days=data.get("waiting_period_days"),
        )


@dataclass
class BuyerResponseData:
    """Structured buyer response."""

    decision: BuyerDecision
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"decision": self.decision.value}
        if self.reason:
            result["reason"] = self.reason
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuyerResponseData":
        """Create from dictionary."""
        return cls(
            decision=BuyerDecision(data["decision"]),
            reason=data.get("reason"),
        )


@dataclass
class CallSession:
    """Represents an active call session."""

    call_id: CallID
    lead_id: LeadID
    started_at: int  # Simulated minute
    ended_at: Optional[int] = None
    duration_minutes: int = 0
    offers_presented: list[PlanOffer] = field(default_factory=list)
    buyer_responses: list[BuyerResponseData] = field(default_factory=list)
    outcome: Optional[BuyerDecision] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "call_id": self.call_id,
            "lead_id": self.lead_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_minutes": self.duration_minutes,
            "offers_presented": [o.to_dict() for o in self.offers_presented],
            "buyer_responses": [r.to_dict() for r in self.buyer_responses],
            "outcome": self.outcome.value if self.outcome else None,
        }


@dataclass
class Appointment:
    """A scheduled appointment."""

    appointment_id: AppointmentID
    lead_id: LeadID
    scheduled_day: int  # Day 1-10
    scheduled_hour: int  # Hour 9-17
    created_at: int  # Simulated minute when created
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "appointment_id": self.appointment_id,
            "lead_id": self.lead_id,
            "scheduled_day": self.scheduled_day,
            "scheduled_hour": self.scheduled_hour,
            "created_at": self.created_at,
            "completed": self.completed,
        }
