"""Pytest fixtures for SalesBench tests."""

import pytest

from salesbench.core.config import BudgetConfig, SalesBenchConfig, ScoringConfig
from salesbench.core.types import (
    BuyerDecision,
    ToolCall,
)
from salesbench.envs.sales_mvp.env import SalesEnv
from salesbench.envs.sales_mvp.personas import Persona, PersonaGenerator
from salesbench.envs.sales_mvp.products import ProductCatalog
from salesbench.orchestrator.orchestrator import Orchestrator


@pytest.fixture
def default_seed() -> int:
    """Default random seed for reproducibility."""
    return 42


@pytest.fixture
def default_config(default_seed: int) -> SalesBenchConfig:
    """Create a default configuration for tests."""
    return SalesBenchConfig(
        seed=default_seed,
        num_leads=10,  # Smaller for faster tests
        budget=BudgetConfig(
            total_days=10,
            max_calls_per_day=50,
            max_tool_calls_per_turn=10,
            max_offers_per_call=3,
        ),
        scoring=ScoringConfig(),
    )


@pytest.fixture
def persona_generator(default_seed: int) -> PersonaGenerator:
    """Create a persona generator with fixed seed."""
    return PersonaGenerator(seed=default_seed)


@pytest.fixture
def sample_personas(persona_generator: PersonaGenerator) -> list[Persona]:
    """Generate a batch of sample personas."""
    return persona_generator.generate_batch(10)


@pytest.fixture
def product_catalog() -> ProductCatalog:
    """Create a product catalog."""
    return ProductCatalog()


@pytest.fixture
def sales_env(default_config: SalesBenchConfig) -> SalesEnv:
    """Create a SalesEnv instance."""
    env = SalesEnv(default_config)
    env.reset()
    return env


@pytest.fixture
def orchestrator(default_config: SalesBenchConfig) -> Orchestrator:
    """Create an Orchestrator instance."""
    return Orchestrator(default_config)


@pytest.fixture
def mock_buyer_simulator():
    """Create a mock buyer simulator that always rejects."""

    def simulator(lead, offer, session, pitch=None, negotiation_history=None):
        from salesbench.core.types import BuyerResponseData

        return BuyerResponseData(
            decision=BuyerDecision.REJECT_PLAN,
            reason="Test rejection",
            dialogue="No thank you.",
        )

    return simulator


@pytest.fixture
def accepting_buyer_simulator():
    """Create a mock buyer simulator that always accepts."""

    def simulator(lead, offer, session, pitch=None, negotiation_history=None):
        from salesbench.core.types import BuyerResponseData

        return BuyerResponseData(
            decision=BuyerDecision.ACCEPT_PLAN,
            reason="Test acceptance",
            dialogue="Yes, I'll take it!",
        )

    return simulator


@pytest.fixture
def sample_tool_call() -> ToolCall:
    """Create a sample tool call."""
    return ToolCall(
        tool_name="crm.search_leads",
        arguments={"temperature": "hot"},
    )


@pytest.fixture
def sample_plan_offer() -> dict:
    """Create a sample plan offer for propose_plan."""
    return {
        "plan_id": "TERM",
        "coverage_amount": 500000,
        "term_years": 20,
        "next_step": "close_now",
    }
