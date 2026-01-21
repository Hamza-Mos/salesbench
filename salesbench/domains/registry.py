"""Domain registration and discovery."""

from typing import Type

from salesbench.domains.base import BaseDomain

_DOMAINS: dict[str, Type[BaseDomain]] = {}


def register_domain(name: str):
    """Decorator to register a domain.

    Usage:
        @register_domain("insurance")
        class InsuranceDomain(BaseDomain):
            ...

    Args:
        name: Unique identifier for the domain

    Returns:
        Decorator function
    """

    def decorator(cls: Type[BaseDomain]):
        if name in _DOMAINS:
            raise ValueError(f"Domain '{name}' is already registered")
        _DOMAINS[name] = cls
        return cls

    return decorator


def get_domain(name: str) -> BaseDomain:
    """Get domain instance by name.

    Args:
        name: Domain identifier (e.g., "insurance")

    Returns:
        Instantiated domain object

    Raises:
        ValueError: If domain not found
    """
    if name not in _DOMAINS:
        available = list(_DOMAINS.keys())
        raise ValueError(f"Unknown domain: '{name}'. Available: {available}")
    return _DOMAINS[name]()


def list_domains() -> list[str]:
    """List available domain names.

    Returns:
        List of registered domain identifiers
    """
    return list(_DOMAINS.keys())
