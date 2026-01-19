"""Verifiers-compatible environment wrapper.

Usage:
    from salesbench import load_environment

    env = load_environment(seed=42, num_leads=100)
    obs = env.reset()

    while not env.is_done:
        tool_calls = agent.generate(obs)
        obs, reward, done, info = env.step(tool_calls)

Note: OPENAI_API_KEY environment variable is required for the LLM buyer simulator.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from salesbench.core.types import ToolCall, ToolResult
from salesbench.core.config import SalesBenchConfig, BudgetConfig
from salesbench.orchestrator.orchestrator import Orchestrator
from salesbench.envs.sales_mvp.verifiers.scoring import calculate_episode_score
from salesbench.envs.sales_mvp.metrics import compute_episode_metrics


@dataclass
class SalesBenchEnvironment:
    """Verifiers-compatible SalesBench environment."""

    config: SalesBenchConfig
    orchestrator: Optional[Orchestrator] = field(default=None, repr=False)
    _last_observation: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.orchestrator = Orchestrator(self.config)

    def reset(self) -> dict[str, Any]:
        """Reset and return initial observation."""
        obs = self.orchestrator.reset()
        self._last_observation = self._enrich_observation(obs)
        return self._last_observation

    def step(
        self,
        tool_calls: list[ToolCall],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Take a step. Returns (observation, reward, done, info)."""
        result = self.orchestrator.step(tool_calls)

        obs = self._enrich_observation(result.observation)
        self._last_observation = obs

        info = {
            "tool_results": [r.to_dict() for r in result.tool_results],
            "turn": self.orchestrator.turn_count,
        }

        if result.terminated:
            final_result = self.orchestrator.get_final_result()
            info["final_result"] = final_result.to_dict()
            info["metrics"] = compute_episode_metrics(
                self.orchestrator.env.state,
                self.config.budget.total_days,
            ).to_dict()
            info["score_breakdown"] = calculate_episode_score(
                self.orchestrator.env.state,
                self.config.scoring,
                self.config.budget.total_days,
            ).to_dict()

        return obs, result.score, result.terminated, info

    def _enrich_observation(self, base_obs: dict[str, Any]) -> dict[str, Any]:
        obs = base_obs.copy()
        obs["tools"] = self.orchestrator.env.get_tools_schema()

        state = self.orchestrator.env.state
        leads_by_temp = {"hot": 0, "warm": 0, "lukewarm": 0, "cold": 0, "hostile": 0}
        for lead in state.leads.values():
            if not lead.on_dnc_list:
                leads_by_temp[lead.temperature.value] += 1

        obs["leads_summary"] = {
            "total": len(state.leads),
            "available": sum(leads_by_temp.values()),
            "by_temperature": leads_by_temp,
        }
        return obs

    def get_tools(self) -> list[dict[str, Any]]:
        """Get available tool schemas."""
        return self.orchestrator.env.get_tools_schema()

    def get_state(self) -> dict[str, Any]:
        """Get current state snapshot."""
        return self.orchestrator.get_state_snapshot()

    def set_buyer_simulator(self, simulator: Callable) -> None:
        """Set a custom buyer simulator."""
        self.orchestrator.set_buyer_simulator(simulator)

    @property
    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.orchestrator.is_terminated


def load_environment(
    seed: int = 42,
    num_leads: int = 100,
    total_days: int = 10,
    max_calls_per_day: int = 50,
    buyer_model: str = "gpt-4o-mini",
    buyer_temperature: float = 0.3,
    api_key: Optional[str] = None,
    debug: bool = False,
) -> SalesBenchEnvironment:
    """Load a SalesBench environment with LLM buyer simulator.

    Args:
        seed: Random seed for reproducibility.
        num_leads: Number of leads to generate.
        total_days: Total simulated business days.
        max_calls_per_day: Maximum calls per day.
        buyer_model: LLM model for buyer decisions.
        buyer_temperature: LLM temperature for buyer (lower = more consistent).
        api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        debug: Enable debug logging.

    Returns:
        SalesBenchEnvironment ready for use.

    Raises:
        ValueError: If no API key is provided or found in environment.

    Example:
        from salesbench import load_environment, ToolCall

        env = load_environment(seed=42)
        obs = env.reset()

        while not env.is_done:
            tool_calls = [ToolCall(tool_name="crm.search_leads", arguments={})]
            obs, reward, done, info = env.step(tool_calls)
    """
    # Get API key
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OpenAI API key is required. Either pass api_key parameter or set "
            "OPENAI_API_KEY environment variable."
        )

    # Override from env vars if set
    buyer_model = os.environ.get("SALESBENCH_BUYER_MODEL", buyer_model)
    buyer_temp_str = os.environ.get("SALESBENCH_BUYER_TEMPERATURE")
    if buyer_temp_str:
        buyer_temperature = float(buyer_temp_str)

    budget = BudgetConfig(
        total_days=total_days,
        max_calls_per_day=max_calls_per_day,
    )

    config = SalesBenchConfig(
        seed=seed,
        num_leads=num_leads,
        budget=budget,
        buyer_model=buyer_model,
        debug=debug,
    )
    config.validate()

    env = SalesBenchEnvironment(config=config)

    # Always set up LLM buyer simulator
    from salesbench.agents.buyer_llm import create_buyer_simulator
    simulator = create_buyer_simulator(
        model=buyer_model,
        temperature=buyer_temperature,
        api_key=resolved_api_key,
    )
    env.set_buyer_simulator(simulator)

    return env
