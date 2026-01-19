"""Base seller agent interface.

Seller agents interact with the environment through tool calls.
They receive tool results and environment state, then decide which
tool(s) to call next.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from salesbench.core.protocol import SellerAction, validate_seller_action
from salesbench.core.types import ToolCall, ToolResult

if TYPE_CHECKING:
    pass


@dataclass
class SellerObservation:
    """Observation provided to the seller agent.

    Contains everything the seller needs to make decisions.
    """

    # Current environment state (what seller can see)
    current_day: int
    current_hour: int
    remaining_minutes: int

    # Recent tool results
    last_tool_results: list[ToolResult] = field(default_factory=list)

    # Context about ongoing call (if any)
    in_call: bool = False
    current_lead_id: Optional[str] = None
    call_duration: int = 0
    offers_this_call: int = 0

    # Summary stats
    total_calls: int = 0
    total_accepts: int = 0
    total_rejects: int = 0
    total_dnc_violations: int = 0

    # Message from environment (errors, warnings, etc.)
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_day": self.current_day,
            "current_hour": self.current_hour,
            "remaining_minutes": self.remaining_minutes,
            "last_tool_results": [r.to_dict() for r in self.last_tool_results],
            "in_call": self.in_call,
            "current_lead_id": self.current_lead_id,
            "call_duration": self.call_duration,
            "offers_this_call": self.offers_this_call,
            "total_calls": self.total_calls,
            "total_accepts": self.total_accepts,
            "total_rejects": self.total_rejects,
            "total_dnc_violations": self.total_dnc_violations,
            "message": self.message,
        }


@dataclass
class SellerConfig:
    """Configuration for seller agents."""

    # Model settings (for LLM-based sellers)
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1000

    # Strategy settings
    max_offers_per_call: int = 3
    min_call_gap_minutes: int = 5
    prefer_hot_leads: bool = True
    aggressive_closing: bool = False

    # Budget awareness
    track_api_costs: bool = True
    max_api_cost_per_episode: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_offers_per_call": self.max_offers_per_call,
            "min_call_gap_minutes": self.min_call_gap_minutes,
            "prefer_hot_leads": self.prefer_hot_leads,
            "aggressive_closing": self.aggressive_closing,
            "track_api_costs": self.track_api_costs,
            "max_api_cost_per_episode": self.max_api_cost_per_episode,
        }


class SellerAgent(ABC):
    """Abstract base class for seller agents.

    Seller agents:
    - Receive observations (environment state + tool results)
    - Output actions containing messages and/or tool calls
    - Messages are free-form text sent to the buyer (the actual conversation)
    - Tool calls are for analytics/operations (CRM, propose_plan for tracking, etc.)
    """

    def __init__(self, config: Optional[SellerConfig] = None):
        """Initialize the seller agent.

        Args:
            config: Agent configuration.
        """
        self.config = config or SellerConfig()
        self._total_api_cost = 0.0
        self._turn_count = 0

    @abstractmethod
    def act(self, observation: SellerObservation) -> SellerAction:
        """Decide what to say and which tools to call based on observation.

        Args:
            observation: Current observation from environment.

        Returns:
            SellerAction containing message and/or tool calls.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for new episode."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dict with agent stats like API cost, turn count.
        """
        return {
            "total_api_cost": self._total_api_cost,
            "turn_count": self._turn_count,
        }

    def _create_action(
        self,
        tool_calls: Optional[list[ToolCall]] = None,
        message: Optional[str] = None,
    ) -> SellerAction:
        """Helper to create and validate an action.

        Args:
            tool_calls: List of tool calls to make (optional).
            message: Free-form text to send to the buyer (optional).

        Returns:
            Validated SellerAction.
        """
        self._turn_count += 1
        action = SellerAction(tool_calls=tool_calls or [], message=message)
        return validate_seller_action(action)

    def _track_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Track API cost for a call.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.

        Returns:
            Cost for this call.
        """
        # Approximate pricing per 1K tokens
        pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        }

        rates = pricing.get(model, {"input": 0.01, "output": 0.03})
        cost = input_tokens / 1000 * rates["input"] + output_tokens / 1000 * rates["output"]

        self._total_api_cost += cost
        return cost


class MultiAgentSeller(SellerAgent):
    """Base class for multi-agent seller architectures.

    Supports hierarchical or ensemble approaches where multiple
    sub-agents collaborate on decisions.
    """

    def __init__(
        self,
        agents: list[SellerAgent],
        config: Optional[SellerConfig] = None,
    ):
        """Initialize multi-agent seller.

        Args:
            agents: List of sub-agents.
            config: Configuration for the orchestrator.
        """
        super().__init__(config)
        self.agents = agents

    @abstractmethod
    def select_agent(self, observation: SellerObservation) -> SellerAgent:
        """Select which agent should act.

        Args:
            observation: Current observation.

        Returns:
            The agent that should handle this turn.
        """
        pass

    def act(self, observation: SellerObservation) -> SellerAction:
        """Delegate to selected agent."""
        agent = self.select_agent(observation)
        return agent.act(observation)

    def reset(self) -> None:
        """Reset all sub-agents."""
        for agent in self.agents:
            agent.reset()
        self._total_api_cost = 0.0
        self._turn_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Aggregate stats from all agents."""
        stats = super().get_stats()
        stats["agent_stats"] = [a.get_stats() for a in self.agents]
        return stats
