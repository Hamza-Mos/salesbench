"""Core types, protocol, and configuration for SalesBench."""

from salesbench.core.types import (
    LeadID,
    CallID,
    ToolCall,
    ToolResult,
    BuyerDecision,
    PlanOffer,
    NextStep,
    PlanType,
    LeadTemperature,
    ObjectionStyle,
    RiskClass,
    CoverageTier,
)
from salesbench.core.protocol import (
    validate_seller_action,
    validate_buyer_response,
    SellerAction,
    BuyerResponse,
)
from salesbench.core.config import (
    SalesBenchConfig,
    BudgetConfig,
    ScoringConfig,
)
from salesbench.core.errors import (
    SalesBenchError,
    ProtocolViolation,
    BudgetExceeded,
    InvalidToolCall,
    InvalidState,
)

__all__ = [
    # Types
    "LeadID",
    "CallID",
    "ToolCall",
    "ToolResult",
    "BuyerDecision",
    "PlanOffer",
    "NextStep",
    "PlanType",
    "LeadTemperature",
    "ObjectionStyle",
    "RiskClass",
    "CoverageTier",
    # Protocol
    "validate_seller_action",
    "validate_buyer_response",
    "SellerAction",
    "BuyerResponse",
    # Config
    "SalesBenchConfig",
    "BudgetConfig",
    "ScoringConfig",
    # Errors
    "SalesBenchError",
    "ProtocolViolation",
    "BudgetExceeded",
    "InvalidToolCall",
    "InvalidState",
]
