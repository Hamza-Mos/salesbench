"""Core types, protocol, and configuration for SalesBench."""

from salesbench.core.config import (
    BudgetConfig,
    SalesBenchConfig,
    ScoringConfig,
)
from salesbench.core.errors import (
    BudgetExceeded,
    InvalidState,
    InvalidToolCall,
    ProtocolViolation,
    SalesBenchError,
)
from salesbench.core.protocol import (
    BuyerResponse,
    SellerAction,
    validate_buyer_response,
    validate_seller_action,
)
from salesbench.core.types import (
    BuyerDecision,
    CallID,
    CoverageTier,
    LeadID,
    LeadTemperature,
    NextStep,
    ObjectionStyle,
    PlanOffer,
    PlanType,
    RiskClass,
    ToolCall,
    ToolResult,
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
