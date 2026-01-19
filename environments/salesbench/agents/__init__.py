"""Agents for SalesBench.

This module provides both buyer and seller agents.
All agents are LLM-based for realistic simulation.

Buyer Agents:
- LLMBuyerSimulator: LLM-based buyer decision making

Seller Agents:
- LLMSellerAgent: LLM-based seller using OpenAI/Anthropic
- StreamingLLMSellerAgent: Streaming variant
- ReActSellerAgent: ReAct-style reasoning agent
"""

# Buyer agents
from salesbench.agents.buyer_llm import (
    LLMBuyerSimulator,
    create_buyer_simulator,
    BuyerSimulatorFn,
)

# Seller base
from salesbench.agents.seller_base import (
    SellerAgent,
    SellerObservation,
    SellerConfig,
    MultiAgentSeller,
)

# LLM sellers
from salesbench.agents.seller_llm import (
    LLMSellerAgent,
    StreamingLLMSellerAgent,
    ReActSellerAgent,
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
]
