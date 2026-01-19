"""Agents for SalesBench.

This module provides both buyer and seller agents.

Buyer Agents:
- LLMBuyerSimulator: LLM-based buyer decision making

Seller Agents:
- LLMSellerAgent: LLM-based seller using OpenAI/Anthropic
- StreamingLLMSellerAgent: Streaming variant
- ReActSellerAgent: ReAct-style reasoning agent
- HeuristicSeller: Deterministic baseline agent (no API calls)
"""

# Buyer agents
from salesbench.agents.buyer_llm import (
    BuyerSimulatorFn,
    LLMBuyerSimulator,
    create_buyer_simulator,
)

# Seller base
from salesbench.agents.seller_base import (
    MultiAgentSeller,
    SellerAgent,
    SellerConfig,
    SellerObservation,
)

# Heuristic sellers
from salesbench.agents.seller_heuristic import (
    HeuristicSeller,
    create_heuristic_seller,
)

# LLM sellers
from salesbench.agents.seller_llm import (
    LLMSellerAgent,
    ReActSellerAgent,
    StreamingLLMSellerAgent,
)

__all__ = [
    # Buyer
    "LLMBuyerSimulator",
    "create_buyer_simulator",
    "BuyerSimulatorFn",
    # Seller base
    "SellerAgent",
    "SellerObservation",
    "SellerConfig",
    "MultiAgentSeller",
    # LLM sellers
    "LLMSellerAgent",
    "StreamingLLMSellerAgent",
    "ReActSellerAgent",
    # Heuristic sellers
    "HeuristicSeller",
    "create_heuristic_seller",
]
