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
from salesbench.runner.results import EpisodeResult, TokenUsage

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
    ) -> EpisodeResult:
        """Run a single episode with full instrumentation.

        Args:
            episode_index: Zero-based episode index.
            seed: Random seed for this episode.
            seller_agent: The seller agent to use.
            buyer_simulator: The buyer simulator function.
            verbose_callback: Optional callback for verbose output.

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
                    seed=seed,
                    seller_agent=seller_agent,
                    buyer_simulator=buyer_simulator,
                    verbose_callback=verbose_callback,
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
        seed: int,
        seller_agent: Any,
        buyer_simulator: Callable,
        verbose_callback: Optional[Callable[[str], None]] = None,
    ):
        """Run the core episode logic.

        Args:
            seed: Random seed for this episode.
            seller_agent: The seller agent.
            buyer_simulator: The buyer simulator.
            verbose_callback: Optional verbose output callback.

        Returns:
            EpisodeResult from orchestrator.
        """
        from salesbench.core.config import SalesBenchConfig
        from salesbench.orchestrator.orchestrator import Orchestrator

        # Create config and orchestrator
        env_config = SalesBenchConfig(
            seed=seed,
            num_leads=self.config.num_leads,
        )
        orchestrator = Orchestrator(
            config=env_config,
            seller_model_spec=self.seller_model_spec,
            buyer_model_spec=self.buyer_model_spec,
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
        while not orchestrator.is_terminated and turn < self.config.max_turns:
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

            # Record seller's message to episode context if present
            if action and action.message:
                orchestrator.record_seller_message(action.message)

            if not tool_calls and not (action and action.message):
                if verbose_callback:
                    verbose_callback(f"  Turn {turn}: No action, ending")
                break

            # Execute step with tool calls
            result = orchestrator.step(tool_calls) if tool_calls else None

            # Verbose output
            if verbose_callback and self.config.verbose:
                if action and action.message:
                    verbose_callback(f"  💬 {action.message[:100]}...")
                for tc in tool_calls:
                    verbose_callback(f"  → {tc.tool_name}({tc.arguments})")
                if result:
                    for tr in result.tool_results:
                        status = "OK" if tr.success else "FAIL"
                        verbose_callback(f"  ← [{status}] {tr.data or tr.error}")

            if result and result.terminated:
                break

            # Update observation for next turn
            if result:
                obs_dict = result.observation
                obs_dict["last_tool_results"] = [tr.to_dict() for tr in result.tool_results]

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

        return SellerObservation(
            current_day=time_info.get("current_day", 1),
            current_hour=time_info.get("current_hour", 9),
            remaining_minutes=time_info.get("remaining_minutes", 480),
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
