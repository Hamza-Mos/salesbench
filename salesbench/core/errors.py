"""Custom exceptions for SalesBench."""

from typing import Any, Optional


class SalesBenchError(Exception):
    """Base exception for all SalesBench errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ProtocolViolation(SalesBenchError):
    """Raised when an agent violates the communication protocol.

    Examples:
    - Seller tries to send a message instead of using tools
    - Buyer tries to use tools instead of returning a decision
    - Invalid tool name or arguments
    """

    pass


class BudgetExceeded(SalesBenchError):
    """Raised when a budget limit is exceeded.

    Examples:
    - Exceeded max calls per day
    - Exceeded max call duration
    - Exceeded total episode time
    """

    def __init__(
        self,
        message: str,
        budget_type: str,
        limit: float,
        current: float,
    ):
        super().__init__(
            message,
            details={
                "budget_type": budget_type,
                "limit": limit,
                "current": current,
            },
        )
        self.budget_type = budget_type
        self.limit = limit
        self.current = current


class InvalidToolCall(SalesBenchError):
    """Raised when a tool call is invalid.

    Examples:
    - Missing required arguments
    - Invalid argument types
    - Tool preconditions not met
    """

    def __init__(
        self,
        message: str,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            details={
                "tool_name": tool_name,
                "arguments": arguments or {},
            },
        )
        self.tool_name = tool_name
        self.arguments = arguments or {}


class InvalidState(SalesBenchError):
    """Raised when the environment is in an invalid state.

    Examples:
    - Trying to end a call when no call is active
    - Trying to start a call with an invalid lead
    - Trying to present an offer outside of a call
    """

    def __init__(
        self,
        message: str,
        current_state: Optional[str] = None,
        expected_state: Optional[str] = None,
    ):
        super().__init__(
            message,
            details={
                "current_state": current_state,
                "expected_state": expected_state,
            },
        )
        self.current_state = current_state
        self.expected_state = expected_state


class LeadNotFound(SalesBenchError):
    """Raised when a lead is not found."""

    def __init__(self, lead_id: str):
        super().__init__(
            f"Lead not found: {lead_id}",
            details={"lead_id": lead_id},
        )
        self.lead_id = lead_id


class CallNotActive(SalesBenchError):
    """Raised when trying to perform call operations without an active call."""

    def __init__(self, operation: str):
        super().__init__(
            f"Cannot {operation}: no active call",
            details={"operation": operation},
        )
        self.operation = operation


class CallAlreadyActive(SalesBenchError):
    """Raised when trying to start a call when one is already active."""

    def __init__(self, current_call_id: str, lead_id: str):
        super().__init__(
            f"Call already active: {current_call_id}",
            details={
                "current_call_id": current_call_id,
                "lead_id": lead_id,
            },
        )
        self.current_call_id = current_call_id
        self.lead_id = lead_id


class DoNotCallViolation(SalesBenchError):
    """Raised when calling a lead on the Do Not Call list."""

    def __init__(self, lead_id: str):
        super().__init__(
            f"Do Not Call violation: lead {lead_id} is on DNC list",
            details={"lead_id": lead_id},
        )
        self.lead_id = lead_id


class EpisodeTerminated(SalesBenchError):
    """Raised when the episode has already terminated."""

    def __init__(self, reason: str):
        super().__init__(
            f"Episode terminated: {reason}",
            details={"reason": reason},
        )
        self.reason = reason
