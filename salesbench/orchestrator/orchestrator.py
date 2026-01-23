"""Main orchestrator for SalesBench episodes.

The orchestrator:
- Manages the turn/state machine
- Routes seller tool calls to the environment
- Tracks budgets and enforces limits
- Checks termination conditions
- Computes scores
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

from salesbench.context.compaction import (
    create_buyer_compaction_fn,
    create_seller_compaction_fn,
)
from salesbench.context.episode import EpisodeContext
from salesbench.core.config import SalesBenchConfig
from salesbench.core.errors import EpisodeTerminated
from salesbench.core.types import ToolCall, ToolResult
from salesbench.envs.sales_mvp.env import SalesEnv
from salesbench.models import ModelSpec, get_model_config
from salesbench.orchestrator.budgets import BudgetTracker
from salesbench.orchestrator.termination import (
    TerminationChecker,
    TerminationStatus,
)


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
        safety_max_turns: Optional[int] = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Configuration for the environment.
            scorer: Optional custom scoring function.
            seller_model_spec: Model specification for seller (used for context management).
            buyer_model_spec: Model specification for buyer (used for context management).
            safety_max_turns: Optional safety ceiling for turns (None = natural termination only).
        """
        self.config = config or SalesBenchConfig()
        self._scorer = scorer
        self._seller_model_spec = seller_model_spec
        self._buyer_model_spec = buyer_model_spec
        self._safety_max_turns = safety_max_turns

        # Create environment
        self._env = SalesEnv(self.config)

        # Create budget tracker
        self._budget_tracker = BudgetTracker(self.config.budget)

        # Create termination checker
        self._termination_checker = TerminationChecker(
            budget=self.config.budget,
            safety_max_turns=safety_max_turns,
        )

        # Episode state
        self._initialized = False
        self._terminated = False
        self._termination_status: Optional[TerminationStatus] = None
        self._history: list[dict[str, Any]] = []
        self._cumulative_score = 0.0

        # Get model configs for context management (use defaults if not provided)
        seller_cfg = seller_model_spec.config if seller_model_spec else get_model_config("gpt-4o")
        buyer_cfg = buyer_model_spec.config if buyer_model_spec else get_model_config("gpt-4o-mini")

        # Create compaction functions using same models as agents
        seller_compaction_fn = None
        buyer_compaction_fn = None

        if seller_model_spec:
            try:
                seller_compaction_fn = create_seller_compaction_fn(
                    seller_model_spec.provider,
                    seller_model_spec.model,
                )
                logger.debug(f"[Orchestrator] Created seller compaction fn: {seller_model_spec}")
            except Exception as e:
                logger.warning(f"[Orchestrator] Failed to create seller compaction fn: {e}")

        if buyer_model_spec:
            try:
                buyer_compaction_fn = create_buyer_compaction_fn(
                    buyer_model_spec.provider,
                    buyer_model_spec.model,
                )
                logger.debug(f"[Orchestrator] Created buyer compaction fn: {buyer_model_spec}")
            except Exception as e:
                logger.warning(f"[Orchestrator] Failed to create buyer compaction fn: {e}")

        # Episode context for conversation management with LLM-based compaction
        self._episode_context = EpisodeContext(
            seller_model_config=seller_cfg,
            buyer_model_config=buyer_cfg,
            seller_compaction_fn=seller_compaction_fn,
            buyer_compaction_fn=buyer_compaction_fn,
        )

        # Guardrails for benchmark-clean trajectories
        self._propose_without_message_count: int = 0

        # Stall detection: track turns since last tool call during active calls
        self._turns_since_tool_call: int = 0

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

        # Get model configs for context management (use defaults if not provided)
        seller_cfg = (
            self._seller_model_spec.config
            if self._seller_model_spec
            else get_model_config("gpt-4o")
        )
        buyer_cfg = (
            self._buyer_model_spec.config
            if self._buyer_model_spec
            else get_model_config("gpt-4o-mini")
        )

        # Create compaction functions using same models as agents
        seller_compaction_fn = None
        buyer_compaction_fn = None

        if self._seller_model_spec:
            try:
                seller_compaction_fn = create_seller_compaction_fn(
                    self._seller_model_spec.provider,
                    self._seller_model_spec.model,
                )
            except Exception as e:
                logger.warning(f"[Orchestrator] Failed to create seller compaction fn: {e}")

        if self._buyer_model_spec:
            try:
                buyer_compaction_fn = create_buyer_compaction_fn(
                    self._buyer_model_spec.provider,
                    self._buyer_model_spec.model,
                )
            except Exception as e:
                logger.warning(f"[Orchestrator] Failed to create buyer compaction fn: {e}")

        # Reset episode context for new episode with LLM-based compaction
        self._episode_context = EpisodeContext(
            seller_model_config=seller_cfg,
            buyer_model_config=buyer_cfg,
            seller_compaction_fn=seller_compaction_fn,
            buyer_compaction_fn=buyer_compaction_fn,
        )

        # Set episode context on environment for buyer history
        self._env.set_episode_context(self._episode_context)

        # Reset guardrails
        self._propose_without_message_count = 0
        self._turns_since_tool_call = 0

        return observation

    def step(self, tool_calls: list[ToolCall], seller_message: Optional[str] = None) -> TurnResult:
        """Process a turn with the given tool calls.

        Args:
            tool_calls: List of tool calls from the seller agent.
            seller_message: Optional message the seller spoke this turn.

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
        proposal_executed = False
        successful_tool_calls = 0
        call_started_this_turn = False
        for tc in tool_calls:
            try:
                # --- Guardrail: require a spoken seller message when proposing a plan ---
                if tc.tool_name == "calling.propose_plan" and not (
                    seller_message and seller_message.strip()
                ):
                    self._propose_without_message_count += 1
                    count = self._propose_without_message_count

                    if count == 1:
                        error_msg = (
                            "Protocol violation: calling.propose_plan requires a seller message. "
                            "You MUST output spoken text (your pitch) alongside the propose_plan tool call. "
                            "The buyer hears your message, not the tool call - propose_plan is for analytics only."
                        )
                    else:
                        warning_text = f"propose_plan called {count}x without message - include your pitch as spoken text"
                        error_msg = (
                            f"REPEATED VIOLATION ({count}x): propose_plan requires spoken text! "
                            f"Output your pitch as a message AND call propose_plan together."
                        )
                        self._episode_context._anchored_state.add_protocol_warning(warning_text)
                        logger.warning(f"[PROTOCOL] {warning_text}")

                    tool_results.append(
                        ToolResult(call_id=tc.call_id, success=False, error=error_msg)
                    )
                    continue

                # --- Guardrail: at most one propose_plan per turn ---
                if tc.tool_name == "calling.propose_plan" and proposal_executed:
                    tool_results.append(
                        ToolResult(
                            call_id=tc.call_id,
                            success=False,
                            error="Protocol violation: only one calling.propose_plan is allowed per turn.",
                        )
                    )
                    continue

                # --- Guardrail: graceful skip for call-state precondition failures ---
                if tc.tool_name == "calling.end_call" and not self._env.is_in_call:
                    skip_result = ToolResult(
                        call_id=tc.call_id,
                        success=True,  # Not a failure, just a no-op
                        data={
                            "skipped": True,
                            "reason": "no_active_call",
                            "message": "Skipped: No active call to end. Start a call first.",
                        },
                    )
                    tool_results.append(skip_result)
                    logger.info(f"[TOOL][SKIP] {tc.tool_name} - no active call")
                    continue

                if tc.tool_name == "calling.propose_plan" and not self._env.is_in_call:
                    skip_result = ToolResult(
                        call_id=tc.call_id,
                        success=True,  # Not a failure, just a no-op
                        data={
                            "skipped": True,
                            "reason": "no_active_call",
                            "message": "Skipped: Cannot propose plan without active call. Use calling.start_call first.",
                        },
                    )
                    tool_results.append(skip_result)
                    logger.info(f"[TOOL][SKIP] {tc.tool_name} - no active call")
                    continue

                # Execute tool
                result = self._env.execute_tool(tc)
                tool_results.append(result)

                # Track successful tool calls for stall detection
                if result.success:
                    successful_tool_calls += 1
                    if tc.tool_name == "calling.start_call":
                        call_started_this_turn = True

                # Record to episode context
                self._record_tool_to_context(tc, result)

                # Record usage
                self._budget_tracker.record_tool_call()

                # Mark that we executed a proposal (even if rejected/accepted)
                if tc.tool_name == "calling.propose_plan":
                    proposal_executed = True
                    # Reset violation counter and clear warnings on successful proposal
                    if result.success:
                        self._propose_without_message_count = 0
                        self._episode_context._anchored_state.clear_protocol_warnings()

                # --- Guardrail: stop executing remaining tools after call ends ---
                # Prevents noisy "No active call" errors from extra tool calls in same turn.
                if tc.tool_name == "calling.propose_plan" and result.success:
                    data = result.data or {}
                    if data.get("call_ended"):
                        break
                if tc.tool_name == "calling.end_call" and result.success:
                    break

            except Exception as e:
                tool_results.append(
                    ToolResult(
                        call_id=tc.call_id,
                        success=False,
                        error=f"Error executing tool: {str(e)}",
                    )
                )

        # --- Stall detection: track turns without tool calls during active calls ---
        if call_started_this_turn:
            # Reset counter when a new call starts
            self._turns_since_tool_call = 0
        elif self._env.is_in_call:
            # During an active call, track turns without successful tool calls
            if successful_tool_calls > 0:
                self._turns_since_tool_call = 0
            else:
                self._turns_since_tool_call += 1
        else:
            # Not in a call, reset counter
            self._turns_since_tool_call = 0

        # End turn and get observation
        observation = self._env.end_turn()
        self._budget_tracker.reset_turn()
        logger.debug(
            f"[TURN:{self._termination_checker.turn_count}] Turn ended - {self._env.state.time.elapsed_hours}h {int(self._env.state.time.elapsed_minutes):02d}m"
        )

        # --- Inject stall warning if threshold exceeded ---
        max_turns = self.config.budget.max_turns_without_tool_call
        if self._env.is_in_call and self._turns_since_tool_call >= max_turns:
            observation["system_warning"] = (
                f"🚨 STALL DETECTED ({self._turns_since_tool_call} turns without tool calls) - MANDATORY ACTION REQUIRED 🚨\n\n"
                "This conversation is going in circles. You MUST end this call NOW.\n\n"
                "REQUIRED ACTION: calling.end_call(reason='lead_not_ready')\n\n"
                "DO NOT: Ask more questions, explain again, or continue pitching.\n"
                "The buyer is not ready. Move on to the next lead."
            )
            logger.warning(
                f"[STALL] Turn {self._termination_checker.turn_count}: {self._turns_since_tool_call} turns without tool call"
            )

        # Update budget tracker with current time
        self._budget_tracker.record_time(
            elapsed_hours=self._env.state.time.elapsed_hours,
            elapsed_minutes=self._env.state.time.elapsed_minutes,
        )

        # Sync action_based_minutes with the environment's time system
        # The time system tracks elapsed minutes based on action costs (start_call, propose_plan, etc.)
        self._budget_tracker.usage.action_based_minutes = float(
            self._budget_tracker.usage.total_elapsed_minutes
        )

        # Sync call stats
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

        # Record history (including seller message for JSON output)
        history_entry = {
            "turn": self._termination_checker.turn_count,
            "seller_message": seller_message,
            "tool_calls": [tc.to_dict() for tc in tool_calls],
            "tool_results": [tr.to_dict() for tr in tool_results],
            "score": turn_score,
        }
        self._history.append(history_entry)

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

        Score = Total Revenue (sum of monthly premiums from accepted plans).
        This is simple, interpretable, and aligned with the business goal.

        Args:
            tool_results: Results from tool executions.

        Returns:
            Revenue earned this turn (monthly premium of accepted plans).
        """
        revenue = 0.0

        for result in tool_results:
            if not result.success:
                continue

            data = result.data or {}

            # Score = revenue from accepted plans
            decision = data.get("decision")
            if decision == "accept_plan":
                offer = data.get("offer_presented", {})
                premium = offer.get("monthly_premium", 0)
                revenue += premium

        return revenue

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

    def record_seller_message(
        self,
        message: str,
        raw_llm_content: Optional[Any] = None,
    ) -> None:
        """Record a seller's spoken message to the episode context.

        This should be called by the executor when the seller agent
        produces a message to be spoken to the buyer.

        Args:
            message: The message the seller is speaking (can be empty for tool-only turns).
            raw_llm_content: Provider-specific content (e.g., Gemini Content with
                            thought_signature) for multi-turn conversation preservation.
                            IMPORTANT: Must be passed even for tool-only turns to
                            preserve Gemini 3's mandatory thought_signature.
        """
        # Only mark as spoken message if there's actual content
        # Tool-only turns still need to record gemini_content for thought_signature
        is_spoken = bool(message and message.strip())
        self._episode_context.record_seller_action(
            content=message if is_spoken else "[tool calls only]",
            is_spoken_message=is_spoken,
            gemini_content=raw_llm_content,
        )

    def get_buyer_response(self, seller_message: str) -> Optional[str]:
        """Get a conversational response from the buyer.

        This is called when the seller speaks while in a call.
        Returns a natural conversational response (not a decision).

        Args:
            seller_message: What the salesperson just said.

        Returns:
            The buyer's dialogue response, or None if not in a call.
        """
        return self._env.get_buyer_conversational_response(seller_message)

    @property
    def is_in_call(self) -> bool:
        """Check if there's an active call."""
        return self._env.is_in_call

    def record_buyer_conversation(self, dialogue: str) -> None:
        """Record a buyer's conversational response to the last history entry.

        This should be called after step() when the buyer responds to the seller
        without making a decision (just conversation).

        Args:
            dialogue: What the buyer said.
        """
        if self._history:
            self._history[-1]["buyer_response"] = dialogue

    def record_conversation_turn(self, tokens: int = 0) -> None:
        """Record a conversation turn during an active call.

        This tracks time cost for the seller-buyer conversation exchange.
        Should be called when the seller speaks and the buyer responds
        during an active call.

        Args:
            tokens: Number of tokens in this turn (for token-based tracking).
        """
        self._budget_tracker.record_conversation_turn(tokens)

    def record_seller_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record seller LLM token usage for time tracking.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        total_tokens = input_tokens + output_tokens
        self._budget_tracker.record_inference(input_tokens, output_tokens)
        self._budget_tracker.record_token_time(total_tokens)

    def record_buyer_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record buyer LLM token usage for time tracking.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        total_tokens = input_tokens + output_tokens
        self._budget_tracker.record_token_time(total_tokens)

    def get_final_result(self) -> EpisodeResult:
        """Get the final result of the episode.

        Returns:
            EpisodeResult with summary and metrics.
        """
        stats = self._env.state.stats
        usage = self._budget_tracker.usage

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
            "hours_used": self._env.state.time.elapsed_hours,
            # Revenue-based score (sum of monthly premiums from accepted plans)
            "total_revenue": self._cumulative_score,
            # Dual time metrics - always output both regardless of time_model
            "action_based_minutes": usage.action_based_minutes,
            "token_based_minutes": usage.token_based_minutes,
            "time_model_used": self.config.budget.time_model,
            "budget_minutes_used": self._budget_tracker.get_budget_minutes(),
            "conversation_turns": usage.conversation_turns,
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
