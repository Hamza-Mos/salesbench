"""SalesBench - AI Social Intelligence Benchmark.

A Prime Intellect Verifiers-compatible benchmark for evaluating AI social
intelligence through life insurance cold-calling scenarios.

Features:
- Deterministic buyer decisions (eliminating LLM buyer bias)
- Seeded persona generation (100 reproducible leads per episode)
- Structured tool-based interactions
- RL-ready scoring rubric
"""

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
)
from salesbench.core.config import SalesBenchConfig
from salesbench.envs.sales_mvp.env import SalesEnv
from salesbench.orchestrator.orchestrator import Orchestrator
from salesbench.environment import load_environment, SalesBenchEnvironment

__version__ = "0.1.0"

__all__ = [
    # Environment loader (main entry point)
    "load_environment",
    "SalesBenchEnvironment",
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
    # Config
    "SalesBenchConfig",
    # Environment
    "SalesEnv",
    # Orchestrator
    "Orchestrator",
    # Version
    "__version__",
]
