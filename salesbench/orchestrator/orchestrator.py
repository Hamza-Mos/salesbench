"""Main orchestrator for SalesBench episodes.

The orchestrator:
- Manages the turn/state machine
- Routes seller tool calls to the environment
- Tracks budgets and enforces limits
- Checks termination conditions
- Computes scores
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from salesbench.context.episode import EpisodeContext
from salesbench.core.config import SalesBenchConfig
from salesbench.core.errors import BudgetExceeded, EpisodeTerminated
from salesbench.core.types import ToolCall, ToolResult
from salesbench.envs.sales_mvp.env import SalesEnv
from salesbench.models import ModelConfig, ModelSpec
from salesbench.orchestrator.budgets import BudgetTracker
from salesbench.orchestrator.termination import TerminationChecker, TerminationStatus


@dataclass
class TurnResult:
    """Result of a single turn."""

    tool_results: list[ToolResult]
    observation: dict[str, Any]
    terminated: bool
    termination_reason: Optional[str] = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_results": [r.to_dict() for r in self.tool_results],
            "observation": self.observation,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "score": self.score,
        }


@dataclass
class EpisodeResult:
    """Result of a complete episode."""

    total_turns: int
    final_score: float
    metrics: dict[str, Any]
    termination_reason: str
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_turns": self.total_turns,
            "final_score": self.final_score,
            "metrics": self.metrics,
            "termination_reason": self.termination_reason,
            "history_length": len(self.history),
        }


class Orchestrator:
    """Main orchestrator for SalesBench episodes.

    The orchestrator manages the episode lifecycle:
    1. reset() - Initialize environment and state
    2. step() - Process seller tool calls, return results
    3. Repeat until terminated
    4. get_final_result() - Get episode summary

    Example:
        orchestrator = Orchestrator(config)
        obs = orchestrator.reset()

        while not orchestrator.is_terminated:
            tool_calls = agent.generate_tool_calls(obs)
            result = orchestrator.step(tool_calls)
            obs = result.observation

        final = orchestrator.get_final_result()
    """

    def __init__(
        self,
        config: Optional[SalesBenchConfig] = None,
        scorer: Optional[Callable[["Orchestrator"], float]] = None,
        seller_model_spec: Optional[ModelSpec] = None,
        buyer_model_spec: Optional[ModelSpec] = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Configuration for the environment.
            scorer: Optional custom scoring function.
            seller_model_spec: Model specification for seller (used for context management).
            buyer_model_spec: Model specification for buyer (used for context management).
        """
        self.config = config or SalesBenchConfig()
        self._scorer = scorer
        self._seller_model_spec = seller_model_spec
        self._buyer_model_spec = buyer_model_spec

        # Create environment
        self._env = SalesEnv(self.config)

        # Create budget tracker
        self._budget_tracker = BudgetTracker(self.config.budget)

        # Create termination checker
        self._termination_checker = TerminationChecker(
            budget=self.config.budget,
            max_turns=None,  # No turn limit by default
        )

        # Episode state
        self._initialized = False
        self._terminated = False
        self._termination_status: Optional[TerminationStatus] = None
        self._history: list[dict[str, Any]] = []
        self._cumulative_score = 0.0

        # Get model configs for context management
        seller_cfg = seller_model_spec.config if seller_model_spec else None
        buyer_cfg = buyer_model_spec.config if buyer_model_spec else None

        # Episode context for conversation management with model-aware compression
        self._episode_context = EpisodeContext(
            seller_model_config=seller_cfg,
            buyer_model_config=buyer_cfg,
        )

    @property
    def is_terminated(self) -> bool:
        """Check if the episode is terminated."""
        return self._terminated

    @property
    def env(self) -> SalesEnv:
        """Get the underlying environment."""
        return self._env

    @property
    def turn_count(self) -> int:
        """Get the current turn count."""
        return self._termination_checker.turn_count

    @property
    def episode_context(self) -> EpisodeContext:
        """Get the episode context for conversation management."""
        return self._episode_context

    def reset(self) -> dict[str, Any]:
        """Reset the orchestrator and environment.

        Returns:
            Initial observation dict.
        """
        # Reset environment
        observation = self._env.reset()

        # Reset trackers
        self._budget_tracker.reset()
        self._termination_checker.reset()

        # Reset episode state
        self._initialized = True
        self._terminated = False
        self._termination_status = None
        self._history = []
        self._cumulative_score = 0.0

        # Get model configs for context management
        seller_cfg = self._seller_model_spec.config if self._seller_model_spec else None
        buyer_cfg = self._buyer_model_spec.config if self._buyer_model_spec else None

        # Reset episode context for new episode with model-aware compression
        self._episode_context = EpisodeContext(
            seller_model_config=seller_cfg,
            buyer_model_config=buyer_cfg,
        )

        # Set episode context on environment for buyer history
        self._env.set_episode_context(self._episode_context)

        return observation

    def step(self, tool_calls: list[ToolCall]) -> TurnResult:
        """Process a turn with the given tool calls.

        Args:
            tool_calls: List of tool calls from the seller agent.

        Returns:
            TurnResult with tool results and new observation.

        Raises:
            EpisodeTerminated: If episode is already terminated.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call reset() first.")

        if self._terminated:
            raise EpisodeTerminated(
                self._termination_status.message if self._termination_status else "Episode ended"
            )

        # Increment turn counter
        self._termination_checker.increment_turn()

        # Execute tool calls
        tool_results = []
        for tc in tool_calls:
            try:
                # Check budget before execution
                self._budget_tracker.enforce_tool_call()

                # Execute tool
                result = self._env.execute_tool(tc)
                tool_results.append(result)

                # Record to episode context
                self._record_tool_to_context(tc, result)

                # Record usage
                self._budget_tracker.record_tool_call()

            except BudgetExceeded as e:
                tool_results.append(
                    ToolResult(
                        call_id=tc.call_id,
                        success=False,
                        error=str(e),
                    )
                )
                break
            except Exception as e:
                tool_results.append(
                    ToolResult(
                        call_id=tc.call_id,
                        success=False,
                        error=f"Error executing tool: {str(e)}",
                    )
                )

        # End turn and get observation
        observation = self._env.end_turn()
        self._budget_tracker.reset_turn()

        # Update budget tracker with current time
        self._budget_tracker.record_time(
            day=self._env.state.time.current_day,
            hour=self._env.state.time.current_hour,
            minute=self._env.state.time.current_minute,
        )

        # Sync call stats
        self._budget_tracker.usage.calls_today = self._env.state.stats.calls_today
        self._budget_tracker.usage.calls_total = self._env.state.stats.total_calls
        self._budget_tracker.usage.call_minutes_total = self._env.state.stats.total_call_minutes

        # Check for termination
        termination_status = self._termination_checker.check_all(self._env.state)
        if termination_status.terminated or self._env.is_terminated:
            self._terminated = True
            self._termination_status = termination_status
            if self._env.termination_reason:
                self._termination_status.message = self._env.termination_reason

        # Calculate turn score
        turn_score = self._calculate_turn_score(tool_results)
        self._cumulative_score += turn_score

        # Record history
        self._history.append(
            {
                "turn": self._termination_checker.turn_count,
                "tool_calls": [tc.to_dict() for tc in tool_calls],
                "tool_results": [tr.to_dict() for tr in tool_results],
                "score": turn_score,
            }
        )

        return TurnResult(
            tool_results=tool_results,
            observation=observation,
            terminated=self._terminated,
            termination_reason=(
                self._termination_status.message if self._termination_status else None
            ),
            score=turn_score,
        )

    def _calculate_turn_score(self, tool_results: list[ToolResult]) -> float:
        """Calculate score for a turn based on tool results.

        Args:
            tool_results: Results from tool executions.

        Returns:
            Score for this turn.
        """
        score = 0.0
        scoring = self.config.scoring

        for result in tool_results:
            if not result.success:
                continue

            data = result.data or {}

            # Check for buyer decisions
            decision = data.get("decision")
            if decision == "accept_plan":
                score += scoring.accept_reward
                # Bonus for close_now
                offer = data.get("offer_presented", {})
                if offer.get("next_step") == "close_now":
                    score += scoring.close_now_bonus
                # Premium-based bonus
                premium = offer.get("monthly_premium", 0)
                score += premium * scoring.premium_multiplier

            elif decision == "reject_plan":
                score += scoring.reject_penalty

            elif decision == "end_call":
                score += scoring.end_call_penalty

            # DNC violation
            if data.get("dnc_violation"):
                score += scoring.dnc_penalty

        return score

    def _record_tool_to_context(self, tool_call: ToolCall, result: ToolResult) -> None:
        """Record a tool call and result to the episode context.

        Inspects the tool call and result to determine what type of event
        occurred and records it appropriately to the episode context.

        Args:
            tool_call: The tool call that was executed.
            result: The result of the tool execution.
        """
        tool_name = tool_call.tool_name
        data = result.data or {}

        # Handle call start
        if tool_name == "calling.start_call" and result.success:
            lead_id = data.get("lead_id")
            if lead_id:
                self._episode_context.record_call_start(lead_id)
            self._episode_context.record_tool_result(tool_name, data, lead_id=lead_id)
            return

        # Handle propose_plan - this is a key event
        if tool_name == "calling.propose_plan" and result.success:
            lead_id = self._episode_context.current_lead_id
            offer = data.get("offer_presented", {})
            decision = data.get("decision")
            dialogue = data.get("dialogue", "")
            reason = data.get("reason", "")

            # Record the offer if present
            if offer and lead_id:
                from salesbench.core.types import NextStep, PlanOffer, PlanType

                try:
                    plan_offer = PlanOffer(
                        plan_id=PlanType(offer.get("plan_id", "TERM")),
                        monthly_premium=offer.get("monthly_premium", 0),
                        coverage_amount=offer.get("coverage_amount", 0),
                        next_step=NextStep(offer.get("next_step", "close_now")),
                        term_years=offer.get("term_years"),
                    )
                    self._episode_context.record_offer(lead_id, plan_offer)
                except (ValueError, KeyError):
                    pass  # Skip if offer data is malformed

            # Record the buyer's decision
            if decision and lead_id:
                self._episode_context.record_buyer_decision(lead_id, decision, dialogue, reason)

            # Check if call ended (accept or end_call decisions end the call)
            if data.get("call_ended") and lead_id:
                reason_str = data.get("message", "Call ended")
                self._episode_context.record_call_end(lead_id, reason_str)

            return

        # Handle end_call
        if tool_name == "calling.end_call" and result.success:
            lead_id = data.get("lead_id") or self._episode_context.current_lead_id
            if lead_id:
                reason_str = data.get("reason", "Seller ended call")
                self._episode_context.record_call_end(lead_id, reason_str)
            self._episode_context.record_tool_result(tool_name, data, lead_id=lead_id)
            return

        # For other tools, just record the result
        lead_id = data.get("lead_id") or self._episode_context.current_lead_id
        self._episode_context.record_tool_result(tool_name, data, lead_id=lead_id)

    def record_seller_message(self, message: str) -> None:
        """Record a seller's spoken message to the episode context.

        This should be called by the executor when the seller agent
        produces a message to be spoken to the buyer.

        Args:
            message: The message the seller is speaking.
        """
        self._episode_context.record_seller_action(
            content=message,
            is_spoken_message=True,
        )

    def get_final_result(self) -> EpisodeResult:
        """Get the final result of the episode.

        Returns:
            EpisodeResult with summary and metrics.
        """
        stats = self._env.state.stats

        metrics = {
            "total_calls": stats.total_calls,
            "total_call_minutes": stats.total_call_minutes,
            "accepted_offers": stats.accepted_offers,
            "rejected_offers": stats.rejected_offers,
            "calls_ended_by_buyer": stats.calls_ended_by_buyer,
            "dnc_violations": stats.dnc_violations,
            "total_tool_calls": self._env.state.total_tool_calls,
            "acceptance_rate": (
                stats.accepted_offers / max(1, stats.accepted_offers + stats.rejected_offers)
            ),
            "days_used": self._env.state.time.current_day,
        }

        return EpisodeResult(
            total_turns=self._termination_checker.turn_count,
            final_score=self._cumulative_score,
            metrics=metrics,
            termination_reason=(
                self._termination_status.message if self._termination_status else "Episode complete"
            ),
            history=self._history,
        )

    def get_state_snapshot(self) -> dict[str, Any]:
        """Get a snapshot of the current state.

        Returns:
            Dict with complete state information.
        """
        return {
            "config": self.config.to_dict(),
            "turn": self._termination_checker.turn_count,
            "terminated": self._terminated,
            "termination_status": (
                self._termination_status.to_dict() if self._termination_status else None
            ),
            "environment": self._env.to_dict(),
            "budget": self._budget_tracker.to_dict(),
            "cumulative_score": self._cumulative_score,
        }

    def set_buyer_simulator(self, simulator: Callable) -> None:
        """Set a custom buyer simulator.

        Args:
            simulator: Buyer simulator function.
        """
        self._env.set_buyer_simulator(simulator)
