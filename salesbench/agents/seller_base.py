"""Base seller agent interface.

Seller agents interact with the environment through tool calls.
They receive tool results and environment state, then decide which
tool(s) to call next.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from salesbench.core.protocol import SellerAction, validate_seller_action
from salesbench.core.types import ToolCall, ToolResult
from salesbench.models import get_model_config, is_supported_model


@dataclass
class SellerObservation:
    """Observation provided to the seller agent.

    Contains everything the seller needs to make decisions.
    """

    # Current environment state (what seller can see)
    elapsed_hours: int
    elapsed_minutes: int
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
            "elapsed_hours": self.elapsed_hours,
            "elapsed_minutes": self.elapsed_minutes,
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

    @classmethod
    def from_dict(cls, obs_dict: dict) -> "SellerObservation":
        """Create SellerObservation from orchestrator dict.

        Args:
            obs_dict: Dictionary from orchestrator with time, call, metrics info.

        Returns:
            SellerObservation instance.
        """
        time_info = obs_dict.get("time", {})
        call_info = obs_dict.get("call", {})
        metrics = obs_dict.get("metrics", {})

        # Convert tool results if present
        tool_results = []
        for tr in obs_dict.get("last_tool_results", []):
            if isinstance(tr, dict):
                tool_results.append(
                    ToolResult(
                        call_id=tr.get("call_id", ""),
                        success=tr.get("success", True),
                        data=tr.get("data"),
                        error=tr.get("error"),
                    )
                )
            else:
                tool_results.append(tr)

        return cls(
            elapsed_hours=time_info.get("elapsed_hours", 0),
            elapsed_minutes=time_info.get("elapsed_minutes", 0),
            remaining_minutes=time_info.get("remaining_minutes", 4800),
            last_tool_results=tool_results,
            in_call=call_info.get("in_call", False),
            current_lead_id=call_info.get("current_lead_id"),
            call_duration=call_info.get("duration", 0),
            offers_this_call=call_info.get("offers_this_call", 0),
            total_calls=metrics.get("total_calls", 0),
            total_accepts=metrics.get("accepted_offers", 0),
            total_rejects=metrics.get("rejected_offers", 0),
            total_dnc_violations=metrics.get("dnc_violations", 0),
            message=obs_dict.get("message"),
        )


@dataclass
class SellerConfig:
    """Configuration for seller agents."""

    # Model settings (for LLM-based sellers)
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 16384  # High default to avoid bottlenecking any model

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
        self._total_input_tokens = 0
        self._total_output_tokens = 0

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
        """Track API cost and token usage for a call.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.

        Returns:
            Cost for this call.
        """
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        # Use centralized pricing from models.py
        if is_supported_model(model):
            config = get_model_config(model)
            if config.input_price_per_million and config.output_price_per_million:
                cost = (
                    input_tokens * config.input_price_per_million / 1_000_000
                    + output_tokens * config.output_price_per_million / 1_000_000
                )
                self._total_api_cost += cost
                return cost

        # Model not in registry - skip cost tracking (no fallback pricing)
        return 0.0

    def get_token_usage(self) -> tuple[int, int]:
        """Get total token usage for this episode.

        Returns:
            Tuple of (input_tokens, output_tokens).
        """
        return self._total_input_tokens, self._total_output_tokens
