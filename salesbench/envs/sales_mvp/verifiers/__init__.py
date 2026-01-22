"""Verifiers integration for SalesBench.

Prime Intellect Verifiers-compatible scoring and verification.

Scoring:
    from salesbench.envs.sales_mvp.verifiers import calculate_episode_revenue
    metrics = calculate_episode_revenue(state)
    score = metrics.total_revenue

Server:
    python -m salesbench.envs.sales_mvp.verifiers.server --port 8000
"""

from salesbench.envs.sales_mvp.verifiers.scoring import (
    RevenueMetrics,
    calculate_episode_revenue,
)

__all__ = [
    "RevenueMetrics",
    "calculate_episode_revenue",
]


def get_verifier_app():
    """Get the FastAPI verifier app for deployment."""
    from salesbench.envs.sales_mvp.verifiers.server import get_app

    return get_app()


def verify_episode(*args, **kwargs):
    """Verify an episode trajectory."""
    from salesbench.envs.sales_mvp.verifiers.server import verify_episode as _verify

    return _verify(*args, **kwargs)
