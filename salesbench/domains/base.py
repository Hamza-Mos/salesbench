"""Abstract base classes for sales domains."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class DomainConfig:
    """Configuration for a sales domain.

    Attributes:
        name: Internal identifier (e.g., "insurance", "retail")
        display_name: Human-readable name (e.g., "Life Insurance Sales")
        description: Brief description of the domain
        product_types: List of product type identifiers
        persona_archetypes: List of buyer personality templates
        tools: List of available tool names
    """

    name: str
    display_name: str
    description: str
    product_types: list[str] = field(default_factory=list)
    persona_archetypes: list[dict[str, Any]] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


class BaseDomain(ABC):
    """Abstract base class for sales domains.

    Each domain defines a complete sales scenario including:
    - Product catalog (what's being sold)
    - Buyer personas (who's being sold to)
    - Available tools (CRM, calling, etc.)
    - Scoring configuration (how success is measured)

    To create a new domain, subclass this and implement all abstract methods.
    Use the @register_domain decorator to make it discoverable.
    """

    @property
    @abstractmethod
    def config(self) -> DomainConfig:
        """Return domain configuration."""
        pass

    @abstractmethod
    def get_product_catalog(self) -> Any:
        """Return domain-specific product catalog.

        Returns:
            Product catalog instance with list_products(), get_product(), etc.
        """
        pass

    @abstractmethod
    def create_persona_generator(self, seed: int) -> Any:
        """Create persona generator for this domain.

        Args:
            seed: Random seed for reproducible persona generation

        Returns:
            PersonaGenerator instance with generate_one(), generate_batch(), etc.
        """
        pass

    @abstractmethod
    def get_tool_handlers(self) -> dict[str, Callable]:
        """Return tool name to handler mapping.

        Returns:
            Dictionary mapping tool names (e.g., "crm.search_leads") to
            handler functions.
        """
        pass

    @abstractmethod
    def get_seller_system_prompt(self) -> str:
        """Return system prompt for the seller agent.

        Returns:
            System prompt string with domain-specific instructions.
        """
        pass

    @abstractmethod
    def get_buyer_system_prompt(self) -> str:
        """Return system prompt for the buyer simulator.

        Returns:
            System prompt string for buyer persona simulation.
        """
        pass

