"""Episode executor for benchmark runs.

Wraps single episode execution with telemetry and storage instrumentation.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from salesbench.models import ModelSpec
from salesbench.runner.config import BenchmarkConfig
from salesbench.runner.integrations import IntegrationManager
from salesbench.runner.results import (
    EpisodeProgress,
    EpisodeResult,
    TokenUsage,
    calculate_cost_breakdown,
)

logger = logging.getLogger(__name__)


class EpisodeExecutor:
    """Executes single episodes with instrumentation.

    Wraps the orchestrator to:
    - Create telemetry spans
    - Write to Supabase
    - Handle errors gracefully
    - Return structured results

    Example:
        executor = EpisodeExecutor(config, integrations)
        result = executor.run_episode(
            episode_index=0,
            seed=42,
            seller_agent=agent,
            buyer_simulator=simulator,
        )
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        integrations: IntegrationManager,
        seller_model_spec: Optional[ModelSpec] = None,
        buyer_model_spec: Optional[ModelSpec] = None,
    ):
        """Initialize the executor.

        Args:
            config: Benchmark configuration.
            integrations: Integration manager for telemetry/storage.
            seller_model_spec: Model specification for seller (for context management).
            buyer_model_spec: Model specification for buyer (for context management).
        """
        self.config = config
        self.integrations = integrations
        self.seller_model_spec = seller_model_spec
        self.buyer_model_spec = buyer_model_spec

    def run_episode(
        self,
        episode_index: int,
        seed: int,
        seller_agent: Any,
        buyer_simulator: Callable,
        verbose_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[EpisodeProgress], None]] = None,
    ) -> EpisodeResult:
        """Run a single episode with full instrumentation.

        Args:
            episode_index: Zero-based episode index.
            seed: Random seed for this episode.
            seller_agent: The seller agent to use.
            buyer_simulator: The buyer simulator function.
            verbose_callback: Optional callback for verbose output.
            progress_callback: Optional callback for real-time progress updates.

        Returns:
            EpisodeResult with metrics and status.
        """
        episode_id = f"{self.config.benchmark_id}_ep{episode_index}_{uuid.uuid4().hex[:6]}"
        model_name = self.config.seller_model or "unknown"

        started_at = datetime.utcnow()
        start_time = time.time()

        result = EpisodeResult(
            episode_id=episode_id,
            benchmark_id=self.config.benchmark_id,
            episode_index=episode_index,
            seed=seed,
            started_at=started_at,
        )

        # Write episode start
        self.integrations.write_episode_start(
            episode_id=episode_id,
            benchmark_id=self.config.benchmark_id,
            seed=seed,
            model_name=model_name,
            num_leads=self.config.num_leads,
            config_dict=self.config.to_dict(),
        )

        span_manager = self.integrations.get_span_manager()

        try:
            with span_manager.episode_span(
                episode_id=episode_id,
                seed=seed,
                model_name=model_name,
                num_leads=self.config.num_leads,
            ) as episode_span:
                # Run the actual episode
                episode_result = self._run_episode_inner(
                    episode_index=episode_index,
                    seed=seed,
                    seller_agent=seller_agent,
                    buyer_simulator=buyer_simulator,
                    verbose_callback=verbose_callback,
                    progress_callback=progress_callback,
                )

                # Update result from orchestrator
                result.final_score = episode_result.final_score
                result.total_turns = episode_result.total_turns
                result.total_accepts = episode_result.metrics.get("accepted_offers", 0)
                result.total_rejects = episode_result.metrics.get("rejected_offers", 0)
                result.total_calls = episode_result.metrics.get("total_calls", 0)
                result.dnc_violations = episode_result.metrics.get("dnc_violations", 0)
                result.termination_reason = episode_result.termination_reason
                result.metrics = episode_result.metrics
                result.trajectory = episode_result.history  # Full conversation log
                result.status = "completed"

                # Track token usage from seller and buyer
                token_usage = TokenUsage()
                if hasattr(seller_agent, "get_token_usage"):
                    seller_input, seller_output = seller_agent.get_token_usage()
                    token_usage.add_seller_usage(seller_input, seller_output)
                if hasattr(buyer_simulator, "get_token_usage"):
                    buyer_input, buyer_output = buyer_simulator.get_token_usage()
                    token_usage.add_buyer_usage(buyer_input, buyer_output)
                result.token_usage = token_usage

                # Calculate cost breakdown based on token usage and model pricing
                seller_model = self.seller_model_spec.model if self.seller_model_spec else ""
                buyer_model = self.buyer_model_spec.model if self.buyer_model_spec else ""
                result.cost_breakdown = calculate_cost_breakdown(
                    token_usage, seller_model, buyer_model
                )

                # Record to span
                episode_span.finish(
                    final_score=result.final_score,
                    metrics=result.metrics,
                )

        except Exception as e:
            logger.error(f"Episode {episode_index} failed: {e}")
            result.status = "failed"
            result.error = str(e)

        # Finalize timing
        result.ended_at = datetime.utcnow()
        result.duration_seconds = time.time() - start_time

        # Write episode end
        self.integrations.write_episode_result(result)

        return result

    def _run_episode_inner(
        self,
        episode_index: int,
        seed: int,
        seller_agent: Any,
        buyer_simulator: Callable,
        verbose_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[EpisodeProgress], None]] = None,
    ):
        """Run the core episode logic.

        Args:
            episode_index: Zero-based episode index.
            seed: Random seed for this episode.
            seller_agent: The seller agent.
            buyer_simulator: The buyer simulator.
            verbose_callback: Optional verbose output callback.
            progress_callback: Optional callback for real-time progress updates.

        Returns:
            EpisodeResult from orchestrator.
        """
        from dataclasses import replace

        from salesbench.orchestrator.orchestrator import Orchestrator

        # Clone episode config with new seed for this specific episode
        # This preserves num_leads, budget, and other settings from the benchmark config
        env_config = replace(self.config.episode_config, seed=seed)

        orchestrator = Orchestrator(
            config=env_config,
            seller_model_spec=self.seller_model_spec,
            buyer_model_spec=self.buyer_model_spec,
            safety_max_turns=self.config.safety_max_turns,
        )
        orchestrator.set_buyer_simulator(buyer_simulator)

        # Reset buyer simulator token tracking if available
        if hasattr(buyer_simulator, "reset"):
            buyer_simulator.reset()

        # Reset and get initial observation
        obs_dict = orchestrator.reset()

        # Get tool schemas
        tool_schemas = orchestrator.env.get_tools_schema()

        turn = 0
        # Loop until natural termination or safety limit (if configured)
        while not orchestrator.is_terminated:
            # Check safety limit if configured
            if self.config.safety_max_turns and turn >= self.config.safety_max_turns:
                break
            turn += 1

            # Convert dict to SellerObservation
            obs = self._dict_to_observation(obs_dict)

            # Get action from agent (returns SellerAction with message + tool_calls)
            # Pass episode context for compressed conversation history
            try:
                action = seller_agent.act(
                    obs,
                    tool_schemas,
                    episode_context=orchestrator.episode_context,
                )
            except Exception as e:
                if verbose_callback:
                    verbose_callback(f"  Agent error: {e}")
                break

            # Extract tool calls from action
            tool_calls = action.tool_calls if action else []
            seller_message = action.message if action else None

            # Record seller's message to episode context if present
            if seller_message:
                orchestrator.record_seller_message(seller_message)

            if not tool_calls and not seller_message:
                if verbose_callback:
                    verbose_callback(f"  Turn {turn}: No action, ending")
                break

            # Execute step with tool calls (pass seller_message for history)
            # Even if no tool calls, we call step to record the message in history
            if tool_calls:
                result = orchestrator.step(tool_calls, seller_message=seller_message)
            elif seller_message:
                # No tool calls but seller sent a message - still record it
                result = orchestrator.step([], seller_message=seller_message)
            else:
                result = None

            # Check if a buyer decision was made (from propose_plan)
            buyer_decision_made = False
            if result:
                for tr in result.tool_results:
                    if tr.data and tr.data.get("decision"):
                        buyer_decision_made = True
                        break

            # Get buyer conversational response if:
            # - Seller sent a message
            # - We're in a call
            # - No decision was made via propose_plan (avoid double response)
            buyer_conversation_response = None
            if seller_message and orchestrator.is_in_call and not buyer_decision_made:
                buyer_conversation_response = orchestrator.get_buyer_response(seller_message)
                if buyer_conversation_response:
                    # Record buyer's response to episode context
                    lead_id = orchestrator.episode_context.current_lead_id
                    if lead_id:
                        orchestrator.episode_context.record_buyer_message(
                            lead_id=lead_id,
                            dialogue=buyer_conversation_response,
                        )
                    # Record conversation turn for time tracking
                    # Estimate tokens from the buyer response length (~4 chars per token)
                    estimated_tokens = len(buyer_conversation_response) // 4
                    orchestrator.record_conversation_turn(tokens=estimated_tokens)

            # Verbose output with clear [SELLER] and [BUYER] labels
            if verbose_callback and self.config.verbose:
                # Seller's spoken message (full, no truncation)
                if seller_message:
                    verbose_callback(f"  [SELLER] {seller_message}")

                # Seller's tool calls
                for tc in tool_calls:
                    verbose_callback(f"  [SELLER][TOOL] {tc.tool_name}({tc.arguments})")

                # Tool results - check for buyer responses
                if result:
                    for tr in result.tool_results:
                        data = tr.data or {}

                        # Determine status: SKIP for graceful skips, OK/FAIL otherwise
                        if data.get("skipped"):
                            status = "SKIP"
                        elif tr.success:
                            status = "OK"
                        else:
                            status = "FAIL"

                        # Check if this contains a buyer response (from propose_plan, etc.)
                        if data.get("dialogue"):
                            decision = data.get("decision", "response")
                            dialogue = data.get("dialogue", "")
                            verbose_callback(f"  [BUYER][{decision.upper()}] {dialogue}")
                        elif data.get("decision"):
                            # Decision without dialogue
                            verbose_callback(f"  [BUYER][{data['decision'].upper()}]")
                        elif data.get("skipped"):
                            # Graceful skip - show reason
                            reason = data.get("reason", "precondition not met")
                            verbose_callback(f"  [TOOL][{status}] {reason}")
                        else:
                            # Regular tool result (full, no truncation)
                            verbose_callback(f"  [TOOL][{status}] {tr.data or tr.error}")

                # Show buyer conversational response (if not already shown via decision)
                if buyer_conversation_response:
                    verbose_callback(f"  [BUYER] {buyer_conversation_response}")

            # Send progress update at configured interval (default every 5 turns)
            interval = self.config.progress_interval
            if progress_callback and (turn == 1 or turn % interval == 0):
                progress = self._create_progress_update(
                    episode_index=episode_index,
                    turn=turn,
                    orchestrator=orchestrator,
                )
                progress_callback(progress)

            if result and result.terminated:
                break

            # Update observation for next turn
            if result:
                obs_dict = result.observation
                obs_dict["last_tool_results"] = [tr.to_dict() for tr in result.tool_results]

            # Add buyer conversational response to observation and history
            if buyer_conversation_response:
                obs_dict["buyer_response"] = buyer_conversation_response
                # Also add as a system message so seller sees it
                obs_dict["message"] = f'Buyer: "{buyer_conversation_response}"'
                # Record in history for JSON output
                orchestrator.record_buyer_conversation(buyer_conversation_response)

        return orchestrator.get_final_result()

    def _dict_to_observation(self, obs_dict: dict):
        """Convert orchestrator dict to SellerObservation.

        Args:
            obs_dict: Observation dict from orchestrator.

        Returns:
            SellerObservation object.
        """
        from salesbench.agents.seller_base import SellerObservation
        from salesbench.core.types import ToolResult

        time_info = obs_dict.get("time", {})
        active_call = obs_dict.get("active_call") or {}
        stats = obs_dict.get("stats", {})

        # Get remaining minutes from the observation or compute from total budget
        remaining_minutes = time_info.get("remaining_minutes", 4800)  # Default 80 hours

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

        # Combine message and system_warning if both present
        message = obs_dict.get("message")
        system_warning = obs_dict.get("system_warning")
        if system_warning:
            if message:
                message = f"{message}\n\n⚠️ {system_warning}"
            else:
                message = f"⚠️ {system_warning}"

        return SellerObservation(
            elapsed_hours=time_info.get("elapsed_hours", 0),
            elapsed_minutes=time_info.get("elapsed_minutes", 0),
            remaining_minutes=remaining_minutes,
            last_tool_results=tool_results,
            in_call=bool(obs_dict.get("has_active_call", False)),
            current_lead_id=(active_call.get("lead_id") if active_call else None),
            call_duration=(active_call.get("duration_minutes", 0) if active_call else 0),
            offers_this_call=(active_call.get("offers_presented", 0) if active_call else 0),
            total_calls=stats.get("total_calls", 0),
            total_accepts=stats.get("accepted_offers", 0),
            total_rejects=stats.get("rejected_offers", 0),
            total_dnc_violations=stats.get("dnc_violations", 0),
            message=message,
        )

    def _create_progress_update(
        self,
        episode_index: int,
        turn: int,
        orchestrator: Any,
    ) -> EpisodeProgress:
        """Create a progress update from current orchestrator state.

        Args:
            episode_index: Zero-based episode index.
            turn: Current turn number.
            orchestrator: The orchestrator instance.

        Returns:
            EpisodeProgress with current state.
        """
        from salesbench.core.types import LeadStatus

        env_state = orchestrator.env.state
        budget = orchestrator.config.budget

        # Count leads by status
        total_leads = len(env_state.leads)
        leads_contacted = 0
        leads_converted = 0
        leads_dnc = 0
        leads_active = 0

        for lead in env_state.leads.values():
            if lead.call_count > 0:
                leads_contacted += 1
            if lead.status == LeadStatus.CONVERTED:
                leads_converted += 1
            elif lead.status == LeadStatus.DNC:
                leads_dnc += 1
            elif lead.status == LeadStatus.ACTIVE:
                leads_active += 1

        return EpisodeProgress(
            episode_index=episode_index,
            turn=turn,
            elapsed_hours=env_state.time.elapsed_hours,
            elapsed_minutes=env_state.time.elapsed_minutes,
            total_hours=budget.total_hours,
            total_leads=total_leads,
            leads_contacted=leads_contacted,
            leads_converted=leads_converted,
            leads_dnc=leads_dnc,
            leads_active=leads_active,
            in_call=env_state.active_call is not None,
            current_lead_id=(
                str(env_state.active_call.lead_id) if env_state.active_call else None
            ),
        )


class AsyncEpisodeExecutor:
    """Async wrapper for episode execution.

    Enables parallel episode execution via asyncio.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        integrations: IntegrationManager,
        seller_model_spec: Optional[ModelSpec] = None,
        buyer_model_spec: Optional[ModelSpec] = None,
    ):
        """Initialize the async executor.

        Args:
            config: Benchmark configuration.
            integrations: Integration manager.
            seller_model_spec: Model specification for seller.
            buyer_model_spec: Model specification for buyer.
        """
        self.executor = EpisodeExecutor(config, integrations, seller_model_spec, buyer_model_spec)
        self.config = config

    async def run_episode(
        self,
        episode_index: int,
        seed: int,
        seller_agent: Any,
        buyer_simulator: Callable,
        verbose_callback: Optional[Callable[[str], None]] = None,
    ) -> EpisodeResult:
        """Run episode asynchronously.

        Actually runs synchronously but in the event loop's executor
        to allow parallel execution.

        Args:
            episode_index: Zero-based episode index.
            seed: Random seed.
            seller_agent: Seller agent.
            buyer_simulator: Buyer simulator.
            verbose_callback: Verbose output callback.

        Returns:
            EpisodeResult.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.executor.run_episode,
            episode_index,
            seed,
            seller_agent,
            buyer_simulator,
            verbose_callback,
        )
