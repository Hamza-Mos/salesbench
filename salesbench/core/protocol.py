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
    - raw_llm_content: Provider-specific content for multi-turn preservation
                       (e.g., Gemini Content with thought_signature)

    The seller can output both a message AND tool calls in the same turn.
    Tool calls like propose_plan are purely analytical and don't affect the conversation.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    message: Optional[str] = None
    raw_llm_content: Optional[Any] = None  # Provider-specific (e.g., Gemini Content)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "type": "seller_action",
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }
        if self.message:
            result["message"] = self.message
        # raw_llm_content is intentionally not serialized - it's for in-memory use only
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
            "description": "Search for leads matching criteria. Returns up to `limit` leads.",
            "properties": {
                "temperature": {
                    "type": "string",
                    "enum": ["hot", "warm", "lukewarm", "cold", "hostile"],
                    "description": "Filter by lead temperature/readiness level.",
                },
                "min_income": {
                    "type": "number",
                    "description": "Minimum annual income filter (e.g., 50000 for $50k+).",
                },
                "max_age": {
                    "type": "integer",
                    "description": "Maximum age filter.",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of leads to return. Default: 10.",
                },
            },
            "required": [],
        },
        "crm.get_lead": {
            "type": "object",
            "description": "Get detailed information about a specific lead.",
            "properties": {
                "lead_id": {
                    "type": "string",
                    "description": "The lead ID (e.g., 'lead_abc123').",
                },
            },
            "required": ["lead_id"],
        },
        "crm.update_lead": {
            "type": "object",
            "description": "Update lead notes or temperature.",
            "properties": {
                "lead_id": {
                    "type": "string",
                    "description": "The lead ID to update.",
                },
                "notes": {
                    "type": "string",
                    "description": "Notes to add to the lead record.",
                },
                "temperature": {
                    "type": "string",
                    "enum": ["hot", "warm", "lukewarm", "cold", "hostile"],
                    "description": "Update the lead's temperature/readiness level.",
                },
            },
            "required": ["lead_id"],
        },
        "crm.log_call": {
            "type": "object",
            "description": "Log a completed call for record-keeping.",
            "properties": {
                "lead_id": {
                    "type": "string",
                    "description": "The lead that was called.",
                },
                "call_id": {
                    "type": "string",
                    "description": "The call ID from start_call.",
                },
                "outcome": {
                    "type": "string",
                    "description": "Call outcome (e.g., 'sale', 'rejected', 'no_answer', 'callback_scheduled').",
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes about the call.",
                },
            },
            "required": ["lead_id", "call_id", "outcome"],
        },
        "calendar.get_availability": {
            "type": "object",
            "description": "Check available time slots. Hours are relative to episode start (0 = start of episode).",
            "properties": {
                "from_hour": {
                    "type": "integer",
                    "description": "Starting hour to check (0-indexed from episode start). Must be within episode's total_hours budget.",
                },
                "num_hours": {
                    "type": "integer",
                    "default": 8,
                    "description": "Number of hours to check from from_hour.",
                },
            },
            "required": [],
        },
        "calendar.schedule_call": {
            "type": "object",
            "description": "Schedule a follow-up call with a lead.",
            "properties": {
                "lead_id": {
                    "type": "string",
                    "description": "The lead ID to schedule a call with.",
                },
                "hour": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Hour to schedule (0-indexed from episode start). Must be in the future and within the episode's total_hours budget. Use calendar.get_availability first to see valid hours.",
                },
            },
            "required": ["lead_id", "hour"],
        },
        "calling.start_call": {
            "type": "object",
            "description": "Start a call with a lead. Must not already be in a call.",
            "properties": {
                "lead_id": {
                    "type": "string",
                    "description": "The lead ID to call (e.g., 'lead_abc123').",
                },
            },
            "required": ["lead_id"],
        },
        "calling.propose_plan": {
            "type": "object",
            "description": "Record a plan offer for analytics. This is purely analytical and does NOT send anything to the buyer. Use your message to actually pitch the plan.",
            "properties": {
                "plan_id": {
                    "type": "string",
                    "enum": ["TERM", "WHOLE", "UL", "VUL", "LTC", "DI"],
                    "description": "The insurance plan type.",
                },
                "monthly_premium": {
                    "type": "number",
                    "description": "Monthly premium amount in dollars. Valid range: 1-10000.",
                },
                "coverage_amount": {
                    "type": "number",
                    "description": "Coverage amount in dollars. For DI, this is the monthly benefit (1000-15000).",
                },
                "next_step": {
                    "type": "string",
                    "enum": ["schedule_followup", "request_info", "close_now"],
                    "description": "Proposed next step after the pitch.",
                },
                "term_years": {
                    "type": "integer",
                    "enum": [10, 15, 20, 30],
                    "description": "Term length for TERM plans only.",
                },
            },
            "required": ["plan_id", "monthly_premium", "coverage_amount", "next_step"],
        },
        "calling.end_call": {
            "type": "object",
            "description": "End the current call. Required after a sale or when done with the lead.",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for ending the call (e.g., 'sale_completed', 'rejected', 'callback_scheduled').",
                },
            },
            "required": [],
        },
        "products.list_plans": {
            "type": "object",
            "description": "List all available insurance plans with their details and coverage ranges.",
            "properties": {},
            "required": [],
        },
        "products.get_plan": {
            "type": "object",
            "description": "Get detailed information about a specific insurance plan.",
            "properties": {
                "plan_id": {
                    "type": "string",
                    "enum": ["TERM", "WHOLE", "UL", "VUL", "LTC", "DI"],
                    "description": "The plan ID to look up.",
                },
            },
            "required": ["plan_id"],
        },
        "products.quote_premium": {
            "type": "object",
            "description": "Get a premium quote for an insurance plan.",
            "properties": {
                "plan_id": {
                    "type": "string",
                    "enum": ["TERM", "WHOLE", "UL", "VUL", "LTC", "DI"],
                    "description": "The insurance plan type.",
                },
                "age": {
                    "type": "integer",
                    "description": (
                        "Age of the insured. Valid ranges by plan: "
                        "TERM/VUL: 18-75. WHOLE/UL: 18-80. LTC: 40-79. DI: 18-60."
                    ),
                },
                "coverage_amount": {
                    "type": "number",
                    "description": (
                        "Coverage amount in dollars. Ranges by plan type: "
                        "TERM: 50000-5000000. "
                        "WHOLE: 25000-10000000. "
                        "UL: 50000-10000000. "
                        "VUL: 100000-25000000. "
                        "LTC: 50000-500000 (benefit pool). "
                        "DI: 1000-15000 (MONTHLY benefit, not annual)."
                    ),
                },
                "risk_class": {
                    "type": "string",
                    "enum": [
                        "preferred_plus",
                        "preferred",
                        "standard_plus",
                        "standard",
                        "substandard",
                    ],
                    "description": "Risk classification. Default: standard_plus.",
                },
                "term_years": {
                    "type": "integer",
                    "enum": [10, 15, 20, 30],
                    "description": "Term length for TERM plans only. Default: 20.",
                },
                "waiting_period_days": {
                    "type": "integer",
                    "enum": [30, 60, 90, 180],
                    "description": "Elimination period before benefits begin. Only for DI and LTC. Default: 90.",
                },
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
