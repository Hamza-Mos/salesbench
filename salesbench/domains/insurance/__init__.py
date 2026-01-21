"""Life insurance sales domain.

This domain wraps the existing sales_mvp implementation and provides
the insurance-specific configuration, products, personas, and prompts.
"""

from typing import Any, Callable

from salesbench.agents.buyer_llm import BUYER_SYSTEM_PROMPT
from salesbench.agents.seller_llm import SELLER_SYSTEM_PROMPT
from salesbench.domains.base import BaseDomain, DomainConfig
from salesbench.domains.registry import register_domain
from salesbench.envs.sales_mvp.personas import ARCHETYPES, PersonaGenerator
from salesbench.envs.sales_mvp.products import ProductCatalog


@register_domain("insurance")
class InsuranceDomain(BaseDomain):
    """Life insurance sales domain.

    Cold-calling benchmark for life insurance agents. Includes:
    - 6 insurance product types (TERM, WHOLE, UL, VUL, LTC, DI)
    - 10 buyer persona archetypes
    - CRM, calling, calendar, and product tools
    - Deterministic pricing based on age, risk class, and coverage
    """

    @property
    def config(self) -> DomainConfig:
        """Return insurance domain configuration."""
        return DomainConfig(
            name="insurance",
            display_name="Life Insurance Sales",
            description="Cold-calling benchmark for life insurance agents. "
            "Agents must search leads, make calls, understand buyer needs, "
            "and propose appropriate insurance plans.",
            product_types=["TERM", "WHOLE", "UL", "VUL", "LTC", "DI"],
            persona_archetypes=[
                {
                    "name": arch["name"],
                    "age_range": arch["age_range"],
                    "income_range": arch["income_range"],
                    "base_trust": arch["base_trust"],
                    "base_interest": arch["base_interest"],
                }
                for arch in ARCHETYPES
            ],
            tools=[
                "crm.search_leads",
                "crm.get_lead",
                "crm.update_lead",
                "crm.log_call",
                "calendar.get_availability",
                "calendar.schedule_call",
                "calling.start_call",
                "calling.propose_plan",
                "calling.end_call",
                "products.list_plans",
                "products.get_plan",
                "products.quote_premium",
            ],
            scoring_config={
                "accept_reward": 100,
                "close_now_bonus": 50,
                "schedule_followup_bonus": 20,
                "reject_penalty": -5,
                "end_call_penalty": -10,
                "dnc_violation_penalty": -200,
                "min_score": -1000,
                "max_score": 10000,
            },
        )

    def get_product_catalog(self) -> ProductCatalog:
        """Return insurance product catalog.

        Returns:
            ProductCatalog with 6 insurance product types.
        """
        return ProductCatalog()

    def create_persona_generator(self, seed: int) -> PersonaGenerator:
        """Create persona generator for insurance leads.

        Args:
            seed: Random seed for reproducible lead generation.

        Returns:
            PersonaGenerator configured for insurance buyer personas.
        """
        return PersonaGenerator(seed=seed)

    def get_tool_handlers(self) -> dict[str, Callable]:
        """Return tool handlers for insurance domain.

        Note: Tool handlers are created per-environment instance.
        This method returns the mapping structure for reference.

        Returns:
            Dictionary describing available tools (handlers are
            instantiated by SalesEnv with state).
        """
        # Note: Actual tool handlers require environment state and are
        # created during SalesEnv initialization. This returns the
        # tool categories for reference.
        return {
            "crm": "CRMTools - Lead management and tracking",
            "calendar": "CalendarTools - Appointment scheduling",
            "calling": "CallingTools - Call management and plan proposals",
            "products": "ProductTools - Product catalog and quotes",
        }

    def get_seller_system_prompt(self) -> str:
        """Return system prompt for insurance seller agent.

        Returns:
            Complete system prompt with insurance-specific instructions.
        """
        return SELLER_SYSTEM_PROMPT

    def get_buyer_system_prompt(self) -> str:
        """Return system prompt for buyer simulator.

        Returns:
            Complete system prompt for simulating insurance buyers.
        """
        return BUYER_SYSTEM_PROMPT

    def get_scoring_config(self) -> dict[str, Any]:
        """Return scoring configuration for insurance sales.

        Returns:
            Dictionary with reward/penalty values for various outcomes.
        """
        return self.config.scoring_config
