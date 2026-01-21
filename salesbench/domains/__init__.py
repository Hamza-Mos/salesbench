"""Domain abstraction layer for SalesBench.

This module provides the infrastructure for multi-domain support,
allowing SalesBench to evaluate agents across different sales scenarios
(insurance, retail, SaaS, etc.).
"""

from salesbench.domains.base import BaseDomain, DomainConfig
from salesbench.domains.registry import get_domain, list_domains, register_domain

__all__ = [
    "BaseDomain",
    "DomainConfig",
    "get_domain",
    "list_domains",
    "register_domain",
]
