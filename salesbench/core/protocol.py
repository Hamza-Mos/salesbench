"""Protocol validators for SalesBench.

Ensures seller only uses tools and buyer only returns structured decisions.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from salesbench.core.errors import ProtocolViolation
from salesbench.core.types import BuyerDecision, ToolCall

# Valid tool names that seller can use
SELLER_TOOLS = frozenset(
    {
        # CRM tools
        "crm.search_leads",
        "crm.get_lead",
        "crm.update_lead",
        "crm.log_call",
        # Calendar tools
        "calendar.get_availability",
        "calendar.schedule_call",
        # Calling tools
        "calling.start_call",
        "calling.propose_plan",
        "calling.end_call",
        # Product tools
        "products.list_plans",
        "products.get_plan",
        "products.quote_premium",
    }
)


@dataclass
class SellerAction:
    """Validated seller action.

    Can contain:
    - message: Free-form text to send to the buyer (the actual conversation)
    - tool_calls: Analytical/operational tools (CRM, products, etc.)

    The seller can output both a message AND tool calls in the same turn.
    Tool calls like propose_plan are purely analytical and don't affect the conversation.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "type": "seller_action",
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }
        if self.message:
            result["message"] = self.message
        return result


@dataclass
class BuyerResponse:
    """Validated buyer response (always a decision)."""

    decision: BuyerDecision
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "type": "buyer_response",
            "decision": self.decision.value,
        }
        if self.reason:
            result["reason"] = self.reason
        return result


def validate_seller_action(action: Union[dict[str, Any], SellerAction]) -> SellerAction:
    """Validate that a seller action contains valid tool calls and/or message.

    The seller can output:
    - A message only (free-form text to the buyer)
    - Tool calls only (analytical/operational)
    - Both message and tool calls

    Args:
        action: The action to validate, either a dict or SellerAction.

    Returns:
        A validated SellerAction.

    Raises:
        ProtocolViolation: If the action is invalid.
    """
    if isinstance(action, SellerAction):
        tool_calls = action.tool_calls
        message = action.message
    elif isinstance(action, dict):
        tool_calls = [
            ToolCall.from_dict(tc) if isinstance(tc, dict) else tc
            for tc in action.get("tool_calls", [])
        ]
        message = action.get("message")
    else:
        raise ProtocolViolation(f"Invalid action type: {type(action)}")

    # Must have at least a message or tool calls
    if not tool_calls and not message:
        raise ProtocolViolation("Seller must provide a message and/or tool calls")

    # Validate tool calls if present
    for tc in tool_calls:
        if tc.tool_name not in SELLER_TOOLS:
            raise ProtocolViolation(
                f"Invalid tool '{tc.tool_name}'. Valid tools: {sorted(SELLER_TOOLS)}"
            )

    return SellerAction(tool_calls=tool_calls, message=message)


def validate_buyer_response(response: Union[dict[str, Any], BuyerResponse]) -> BuyerResponse:
    """Validate that a buyer response is a valid decision.

    Args:
        response: The response to validate, either a dict or BuyerResponse.

    Returns:
        A validated BuyerResponse.

    Raises:
        ProtocolViolation: If the response is invalid.
    """
    if isinstance(response, BuyerResponse):
        return response

    if not isinstance(response, dict):
        raise ProtocolViolation(f"Invalid response type: {type(response)}")

    if "decision" not in response:
        raise ProtocolViolation("Buyer response must contain 'decision'")

    try:
        decision = BuyerDecision(response["decision"])
    except ValueError:
        valid_decisions = [d.value for d in BuyerDecision]
        raise ProtocolViolation(
            f"Invalid decision '{response['decision']}'. Valid: {valid_decisions}"
        )

    return BuyerResponse(
        decision=decision,
        reason=response.get("reason"),
    )


def get_tool_schema(tool_name: str) -> dict[str, Any]:
    """Get the JSON schema for a tool's arguments.

    Args:
        tool_name: The name of the tool.

    Returns:
        JSON schema dict for the tool's arguments.
    """
    schemas = {
        "crm.search_leads": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "string",
                    "enum": ["hot", "warm", "lukewarm", "cold", "hostile"],
                },
                "min_income": {"type": "number"},
                "max_age": {"type": "integer"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": [],
        },
        "crm.get_lead": {
            "type": "object",
            "properties": {
                "lead_id": {"type": "string"},
            },
            "required": ["lead_id"],
        },
        "crm.update_lead": {
            "type": "object",
            "properties": {
                "lead_id": {"type": "string"},
                "notes": {"type": "string"},
                "temperature": {"type": "string"},
            },
            "required": ["lead_id"],
        },
        "crm.log_call": {
            "type": "object",
            "properties": {
                "lead_id": {"type": "string"},
                "call_id": {"type": "string"},
                "outcome": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["lead_id", "call_id", "outcome"],
        },
        "calendar.get_availability": {
            "type": "object",
            "properties": {
                "day": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": [],
        },
        "calendar.schedule_call": {
            "type": "object",
            "properties": {
                "lead_id": {"type": "string"},
                "day": {"type": "integer", "minimum": 1, "maximum": 10},
                "hour": {"type": "integer", "minimum": 9, "maximum": 17},
            },
            "required": ["lead_id", "day", "hour"],
        },
        "calling.start_call": {
            "type": "object",
            "properties": {
                "lead_id": {"type": "string"},
            },
            "required": ["lead_id"],
        },
        "calling.propose_plan": {
            "type": "object",
            "description": "Record a plan offer for analytics. This is purely analytical and does NOT send anything to the buyer. Use your message to actually pitch the plan.",
            "properties": {
                "plan_id": {"type": "string", "enum": ["TERM", "WHOLE", "UL", "VUL", "LTC", "DI"]},
                "monthly_premium": {"type": "number"},
                "coverage_amount": {"type": "number"},
                "next_step": {
                    "type": "string",
                    "enum": ["schedule_followup", "request_info", "close_now"],
                },
                "term_years": {"type": "integer"},
            },
            "required": ["plan_id", "monthly_premium", "coverage_amount", "next_step"],
        },
        "calling.end_call": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
            },
            "required": [],
        },
        "products.list_plans": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "products.get_plan": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
            },
            "required": ["plan_id"],
        },
        "products.quote_premium": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
                "age": {"type": "integer"},
                "coverage_amount": {"type": "number"},
                "risk_class": {"type": "string"},
                "term_years": {"type": "integer"},
            },
            "required": ["plan_id", "age", "coverage_amount"],
        },
    }
    return schemas.get(tool_name, {"type": "object", "properties": {}})


def get_all_tool_schemas() -> dict[str, dict[str, Any]]:
    """Get schemas for all seller tools.

    Returns:
        Dict mapping tool names to their JSON schemas.
    """
    return {tool: get_tool_schema(tool) for tool in SELLER_TOOLS}
