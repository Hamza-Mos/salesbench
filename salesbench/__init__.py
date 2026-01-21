"""SalesBench - AI Social Intelligence Benchmark.

A Prime Intellect Verifiers-compatible benchmark for evaluating AI social
intelligence through life insurance cold-calling scenarios.

Features:
- Verifiers-compatible StatefulToolEnv for Prime Intellect Environments Hub
- Seeded persona generation (reproducible leads per episode)
- Structured tool-based interactions
- RL-ready scoring rubric

Usage:
    # Via verifiers (recommended)
    import verifiers as vf
    env = vf.load_environment("salesbench", seed=42)
    results = env.evaluate(client, model="gpt-4o")

    # Direct import
    from salesbench import load_environment
    env = load_environment(seed=42, num_leads=100)
"""

from salesbench.core.config import SalesBenchConfig
from salesbench.core.types import (
    BuyerDecision,
    CallID,
    LeadID,
    LeadTemperature,
    NextStep,
    PlanOffer,
    PlanType,
    ToolCall,
    ToolResult,
)
from salesbench.envs.sales_mvp.env import SalesEnv

# Optional: verifiers-compatible environment is only available when optional
# dependencies (e.g. `verifiers`, `datasets`) are installed.
try:
    from salesbench.environment import (
        SalesBenchToolEnv,
        create_salesbench_dataset,
        load_environment,
    )
except Exception:  # pragma: no cover
    SalesBenchToolEnv = None  # type: ignore[assignment]
    create_salesbench_dataset = None  # type: ignore[assignment]

    def load_environment(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError(
            "salesbench.load_environment requires optional dependencies. "
            "Install the verifiers stack (e.g. `verifiers`, `datasets`) to use it."
        )


__version__ = "0.1.0"

__all__ = [
    # Verifiers-compatible environment (main entry point)
    "load_environment",
    "SalesBenchToolEnv",
    "create_salesbench_dataset",
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
    # Core simulation
    "SalesEnv",
    # Version
    "__version__",
]
