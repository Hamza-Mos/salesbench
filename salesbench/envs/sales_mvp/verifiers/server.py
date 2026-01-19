"""Verifier HTTP server for Prime Intellect integration.

Provides a FastAPI server that validates episode results and returns scores.
Can be deployed as a standalone service or used for local testing.

Usage:
    # Start server
    python -m salesbench.envs.sales_mvp.verifiers.server

    # Or with uvicorn
    uvicorn salesbench.envs.sales_mvp.verifiers.server:app --port 8000
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy import FastAPI to make it optional
app = None


def create_app():
    """Create FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI required for verifier server. Install: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="SalesBench Verifier",
        description="Prime Intellect Verifier API for SalesBench benchmark",
        version="0.1.0",
    )

    class VerifyRequest(BaseModel):
        """Request to verify an episode."""

        episode_id: str
        seed: int
        trajectory: list[dict[str, Any]]
        final_state: Optional[dict[str, Any]] = None
        config: Optional[dict[str, Any]] = None

    class ScoreResponse(BaseModel):
        """Score response from verifier."""

        episode_id: str
        valid: bool
        score: float
        passed: bool
        components: dict[str, Any]
        metrics: dict[str, Any]
        error: Optional[str] = None

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "salesbench-verifier"}

    @app.get("/info")
    async def info():
        """Get verifier information."""
        return {
            "name": "salesbench",
            "version": "0.1.0",
            "description": "AI Social Intelligence Benchmark - Life Insurance Cold-Calling",
            "scoring": {
                "min_score": -10000,
                "max_score": 100000,
                "pass_threshold": 0,
                "primary_metric": "total_score",
            },
        }

    @app.post("/verify", response_model=ScoreResponse)
    async def verify(request: VerifyRequest):
        """Verify an episode and return score.

        This endpoint replays the trajectory to compute the final score.
        """
        try:
            result = verify_episode(
                episode_id=request.episode_id,
                seed=request.seed,
                trajectory=request.trajectory,
                final_state=request.final_state,
                config=request.config,
            )
            return result
        except Exception as e:
            logger.exception(f"Verification failed for {request.episode_id}")
            return ScoreResponse(
                episode_id=request.episode_id,
                valid=False,
                score=0.0,
                passed=False,
                components={},
                metrics={},
                error=str(e),
            )

    @app.post("/score")
    async def score_state(state: dict[str, Any]):
        """Score a final state directly (without replay).

        Used when the full state is available and trusted.
        """
        try:
            # Reconstruct state (simplified - would need proper deserialization)
            # For now, expect pre-computed metrics
            metrics = state.get("metrics", {})
            score_data = state.get("score", {})

            return {
                "valid": True,
                "score": score_data.get("total_score", 0),
                "components": score_data.get("components", {}),
                "metrics": metrics,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app


def verify_episode(
    episode_id: str,
    seed: int,
    trajectory: list[dict[str, Any]],
    final_state: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Verify an episode by replaying the trajectory.

    Args:
        episode_id: Unique episode identifier.
        seed: Random seed used for the episode.
        trajectory: List of (action, observation, reward) tuples.
        final_state: Optional final state for validation.
        config: Optional configuration overrides.

    Returns:
        Dictionary with verification results.
    """
    from salesbench import load_environment
    from salesbench.core.types import ToolCall

    # Load environment with same seed
    env_config = config or {}
    env = load_environment(
        seed=seed,
        num_leads=env_config.get("num_leads", 100),
        total_days=env_config.get("total_days", 10),
    )

    # Replay trajectory
    obs = env.reset()
    total_reward = 0.0
    replayed_rewards = []

    for step_data in trajectory:
        if env.is_done:
            break

        # Extract tool calls from step
        actions = step_data.get("actions", [])
        tool_calls = []
        for action in actions:
            tool_calls.append(
                ToolCall(
                    tool_name=action.get("tool_name", ""),
                    arguments=action.get("arguments", {}),
                    call_id=action.get("call_id"),
                )
            )

        if not tool_calls:
            continue

        # Execute step
        obs, reward, done, info = env.step(tool_calls)
        total_reward += reward
        replayed_rewards.append(reward)

        if done:
            break

    # Get final results
    if env.is_done:
        final_result = env.orchestrator.get_final_result()
        state = env.orchestrator.env.state

        from salesbench.envs.sales_mvp.verifiers.scoring import calculate_episode_score

        score_components = calculate_episode_score(state)

        metrics = {
            "total_turns": final_result.total_turns,
            "total_calls": final_result.metrics.get("total_calls", 0),
            "accepted_offers": final_result.metrics.get("accepted_offers", 0),
            "rejected_offers": final_result.metrics.get("rejected_offers", 0),
            "dnc_violations": final_result.metrics.get("dnc_violations", 0),
        }

        return {
            "episode_id": episode_id,
            "valid": True,
            "score": score_components.total_score,
            "passed": score_components.num_accepts > 0,
            "components": score_components.to_dict()["components"],
            "metrics": metrics,
            "error": None,
        }
    else:
        # Episode didn't complete
        return {
            "episode_id": episode_id,
            "valid": False,
            "score": 0.0,
            "passed": False,
            "components": {},
            "metrics": {"error": "Episode did not complete"},
            "error": "Trajectory replay did not reach terminal state",
        }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the verifier server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required. Install: pip install uvicorn")

    global app
    if app is None:
        app = create_app()

    uvicorn.run(app, host=host, port=port)


# Create app for direct uvicorn usage
def get_app():
    """Get or create the FastAPI app."""
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SalesBench Verifier Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
