"""Verifiers integration for SalesBench.

Prime Intellect Verifiers-compatible scoring and verification.

Scoring:
    from salesbench.envs.sales_mvp.verifiers import calculate_episode_score
    score = calculate_episode_score(state)

Server:
    python -m salesbench.envs.sales_mvp.verifiers.server --port 8000
"""

from salesbench.envs.sales_mvp.verifiers.scoring import (
    ScoreComponents,
    ScoringRubric,
    calculate_episode_score,
)

__all__ = [
    "ScoringRubric",
    "ScoreComponents",
    "calculate_episode_score",
]


def get_verifier_app():
    """Get the FastAPI verifier app for deployment."""
    from salesbench.envs.sales_mvp.verifiers.server import get_app

    return get_app()


def verify_episode(*args, **kwargs):
    """Verify an episode trajectory."""
    from salesbench.envs.sales_mvp.verifiers.server import verify_episode as _verify

    return _verify(*args, **kwargs)
