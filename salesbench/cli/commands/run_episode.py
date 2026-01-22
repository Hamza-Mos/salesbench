"""Run episode command for CLI.

Runs a single episode with an LLM seller agent.
"""

import argparse
import json
import os
import uuid
from contextlib import nullcontext
from typing import Optional

from salesbench.cli.commands import register_command
from salesbench.cli.telemetry_utils import get_tracer, telemetry_context


class EpisodeRunner:
    """Encapsulates episode running logic."""

    def __init__(
        self,
        args: argparse.Namespace,
        seller_spec,
        buyer_spec,
        telemetry_manager=None,
    ):
        self.args = args
        self.seller_spec = seller_spec
        self.buyer_spec = buyer_spec
        self.telemetry_manager = telemetry_manager
        self.tracer = get_tracer(telemetry_manager)

        # Will be set during setup
        self.orchestrator = None
        self.seller_agent = None
        self.buyer_simulator = None
        self.tool_schemas = None
        self.obs_dict = None

    def setup(self) -> bool:
        """Set up the episode. Returns True on success."""
        from salesbench.agents.buyer_llm import create_buyer_simulator
        from salesbench.agents.seller_llm import LLMSellerAgent
        from salesbench.core.config import SalesBenchConfig
        from salesbench.orchestrator.orchestrator import Orchestrator

        config = SalesBenchConfig(seed=self.args.seed, num_leads=self.args.leads)
        self.orchestrator = Orchestrator(config)

        # Set up LLM buyer simulator
        buyer_temp = float(os.environ.get("SALESBENCH_BUYER_TEMPERATURE", "0.0"))
        self.buyer_simulator = create_buyer_simulator(
            provider=self.buyer_spec.provider,
            model=self.buyer_spec.model,
            temperature=buyer_temp,
        )
        self.orchestrator.set_buyer_simulator(self.buyer_simulator)

        # Set up LLM seller agent
        self.seller_agent = LLMSellerAgent(
            provider=self.seller_spec.provider,
            model=self.seller_spec.model,
        )

        print(f"Starting episode with seed {self.args.seed}, {self.args.leads} leads")
        print(f"Seller: {self.seller_spec}")
        print(f"Buyer: {self.buyer_spec}")
        print("-" * 60)

        self.obs_dict = self.orchestrator.reset()
        print(f"Time: {self.obs_dict['time']['elapsed_hours']}h {self.obs_dict['time']['elapsed_minutes']}m elapsed")
        print(f"Leads available: {self.obs_dict['leads_count']}")

        self.tool_schemas = self.orchestrator.env.get_tools_schema()
        return True

    def run_turn(self, turn: int, turn_span) -> tuple[bool, Optional[dict]]:
        """Run a single turn. Returns (continue, result)."""
        from salesbench.agents.seller_base import SellerObservation
        obs = SellerObservation.from_dict(self.obs_dict)

        # Build prompt summary for logging
        prompt_summary = f"Time: {obs.elapsed_hours}h {obs.elapsed_minutes}m, "
        prompt_summary += f"Calls: {obs.total_calls}, Accepts: {obs.total_accepts}"
        if obs.in_call:
            prompt_summary += f" | IN CALL with {obs.current_lead_id}"

        # LLM agent generates action
        try:
            llm_context = self.tracer.start_as_current_span("llm_call") if self.tracer else nullcontext()
            with llm_context as llm_span:
                if llm_span:
                    llm_span.set_attribute("llm.model", str(self.seller_spec))
                    llm_span.add_event("llm_input", {
                        "prompt_summary": prompt_summary,
                        "in_call": obs.in_call,
                        "current_lead": obs.current_lead_id or "",
                        "elapsed_hours": obs.elapsed_hours,
                        "elapsed_minutes": obs.elapsed_minutes,
                    })

                action = self.seller_agent.act(
                    obs,
                    self.tool_schemas,
                    episode_context=self.orchestrator.episode_context,
                )

                if llm_span and action:
                    llm_span.set_attribute("llm.has_message", bool(action.message))
                    llm_span.set_attribute("llm.tool_count", len(action.tool_calls))

                    response_text = action.message or ""
                    tool_calls_str = ", ".join([
                        f"{tc.tool_name}({json.dumps(tc.arguments)})"
                        for tc in action.tool_calls
                    ]) if action.tool_calls else ""

                    llm_span.add_event("llm_output", {
                        "response_message": response_text[:2000] if response_text else "",
                        "tool_calls": tool_calls_str[:1000] if tool_calls_str else "",
                    })

                    if response_text:
                        llm_span.set_attribute("llm.response", response_text[:4000])
                    if tool_calls_str:
                        llm_span.set_attribute("llm.tool_calls", tool_calls_str[:2000])

        except Exception as e:
            print(f"\nError from LLM seller: {e}")
            if turn_span:
                turn_span.set_attribute("turn.error", str(e))
            return False, None

        tool_calls = action.tool_calls if action else []
        message = action.message if action else None

        if turn_span and message:
            turn_span.set_attribute("seller.message", message[:2000] if len(message) > 2000 else message)
            turn_span.add_event("seller_message", {"message": message[:500]})

        if not tool_calls and not message:
            print(f"\nTurn {turn}: Agent returned no action, ending episode.")
            if turn_span:
                turn_span.add_event("no_action", {"reason": "agent returned empty"})
            return False, None

        if turn_span and tool_calls:
            tool_names = [tc.tool_name for tc in tool_calls]
            turn_span.set_attribute("tools.called", ",".join(tool_names))

        # Step environment
        result = self.orchestrator.step(tool_calls) if tool_calls else None

        # Process results and log to span
        if result:
            self._log_tool_results(turn_span, result)

        if self.args.verbose:
            self._print_verbose_turn(turn, message, tool_calls, result)

        if result and result.terminated:
            print(f"\nEpisode terminated: {result.termination_reason}")
            if turn_span:
                turn_span.add_event("episode_terminated", {"reason": result.termination_reason})
            return False, result

        if result:
            self.obs_dict = result.observation
            self.obs_dict["last_tool_results"] = [tr.to_dict() for tr in result.tool_results]

        return True, result

    def _log_tool_results(self, turn_span, result) -> None:
        """Log tool results to span."""
        if not turn_span:
            return

        for tr in result.tool_results:
            event_attrs = {
                "tool": tr.call_id or "unknown",
                "success": tr.success,
            }
            if tr.error:
                event_attrs["error"] = tr.error
            if tr.data:
                if "decision" in tr.data:
                    decision = tr.data["decision"]
                    event_attrs["buyer_decision"] = decision
                    if decision == "accept_plan":
                        turn_span.add_event("plan_accepted", {
                            "plan_id": tr.data.get("plan_id", ""),
                            "premium": tr.data.get("premium", 0),
                        })
                    elif decision in ["reject_plan", "hang_up"]:
                        turn_span.add_event("plan_rejected", {
                            "reason": tr.data.get("reason", ""),
                        })
                if "dialogue" in tr.data and tr.data["dialogue"]:
                    buyer_text = tr.data["dialogue"]
                    event_attrs["buyer_dialogue"] = buyer_text[:500]
                    turn_span.set_attribute("buyer.response", buyer_text[:2000])
                if "seller_pitch" in tr.data and tr.data["seller_pitch"]:
                    event_attrs["seller_pitch"] = tr.data["seller_pitch"][:500]
            turn_span.add_event("tool_result", event_attrs)

    def _print_verbose_turn(self, turn, message, tool_calls, result) -> None:
        """Print verbose turn output."""
        print(f"\nTurn {turn}:")
        if message:
            print(f"  [speaker] {message[:200]}{'...' if len(message) > 200 else ''}")
        for tc in tool_calls:
            print(f"  -> {tc.tool_name}({tc.arguments})")
        if result:
            for tr in result.tool_results:
                status = "OK" if tr.success else "FAIL"
                print(f"  <- [{status}] {tr.data or tr.error}")
                if tr.success and tr.data:
                    if "seller_pitch" in tr.data and tr.data["seller_pitch"]:
                        print(f"  [seller]  Seller: \"{tr.data['seller_pitch']}\"")
                    if "dialogue" in tr.data and tr.data["dialogue"]:
                        print(f"  [phone] Buyer: \"{tr.data['dialogue']}\"")

    def run_episode_loop(self, episode_span) -> int:
        """Run the main episode loop. Returns turn count."""
        turn = 0

        if episode_span:
            episode_id = f"episode_{uuid.uuid4().hex[:8]}"
            episode_span.set_attribute("episode.id", episode_id)
            episode_span.set_attribute("episode.seed", self.args.seed)
            episode_span.set_attribute("episode.num_leads", self.args.leads)
            episode_span.set_attribute("model.seller", str(self.seller_spec))
            episode_span.set_attribute("model.buyer", str(self.buyer_spec))

        while not self.orchestrator.is_terminated:
            if self.args.max_turns and turn >= self.args.max_turns:
                break
            turn += 1

            turn_context = self.tracer.start_as_current_span(f"turn_{turn}") if self.tracer else nullcontext()
            with turn_context as turn_span:
                if turn_span:
                    turn_span.set_attribute("turn.number", turn)

                should_continue, _ = self.run_turn(turn, turn_span)
                if not should_continue:
                    break

        return turn

    def print_results(self, final) -> None:
        """Print episode results."""
        print("\n" + "=" * 60)
        print("Episode Complete")
        print("=" * 60)
        print(f"  Total turns: {final.total_turns}")
        print(f"  Final score: {final.final_score:.2f}")
        print(f"  Accepts: {final.metrics['accepted_offers']}")
        print(f"  Rejects: {final.metrics['rejected_offers']}")
        print(f"  Calls ended by buyer: {final.metrics['calls_ended_by_buyer']}")
        print(f"  DNC violations: {final.metrics['dnc_violations']}")

        # Print token usage
        print("\nToken Usage:")
        seller_in, seller_out = self.seller_agent.get_token_usage()
        buyer_in, buyer_out = 0, 0
        print(f"  Seller: {seller_in:,} input, {seller_out:,} output")
        if hasattr(self.buyer_simulator, "get_token_usage"):
            buyer_in, buyer_out = self.buyer_simulator.get_token_usage()
            print(f"  Buyer:  {buyer_in:,} input, {buyer_out:,} output")
        print(f"  Total:  {seller_in + buyer_in:,} input, {seller_out + buyer_out:,} output")

    def save_results(self, final) -> str:
        """Save results to disk. Returns results directory path."""
        from salesbench.runner.results import TokenUsage
        from salesbench.storage.json_writer import JSONResultsWriter

        seller_in, seller_out = self.seller_agent.get_token_usage()
        buyer_in, buyer_out = 0, 0
        if hasattr(self.buyer_simulator, "get_token_usage"):
            buyer_in, buyer_out = self.buyer_simulator.get_token_usage()

        token_usage = TokenUsage()
        token_usage.add_seller_usage(seller_in, seller_out)
        token_usage.add_buyer_usage(buyer_in, buyer_out)

        episode_result = {
            "mode": "single_episode",
            "config": {
                "seed": self.args.seed,
                "num_leads": self.args.leads,
                "max_turns": self.args.max_turns,
                "seller_model": str(self.seller_spec),
                "buyer_model": str(self.buyer_spec),
            },
            "episode_results": [
                {
                    "episode_id": f"ep_{self.args.seed}",
                    "episode_index": 0,
                    "seed": self.args.seed,
                    "status": "completed",
                    "final_score": final.final_score,
                    "total_turns": final.total_turns,
                    "total_accepts": final.metrics.get("accepted_offers", 0),
                    "total_rejects": final.metrics.get("rejected_offers", 0),
                    "total_calls": final.metrics.get("total_calls", 0),
                    "dnc_violations": final.metrics.get("dnc_violations", 0),
                    "metrics": final.metrics,
                    "trajectory": final.history if self.args.verbose else [],
                    "token_usage": token_usage.to_dict(),
                }
            ],
            "aggregate_metrics": {
                "total_token_usage": token_usage.to_dict(),
            },
        }

        model_name = self.seller_spec.model
        writer = JSONResultsWriter("results")
        results_dir = writer.write_benchmark(
            benchmark_id=f"episode_{self.args.seed}",
            result=episode_result,
            include_traces=self.args.verbose,
            custom_name=f"episode_{model_name}",
        )

        print(f"\nResults saved to: {results_dir}/")
        print("  - summary.json")
        if self.args.verbose:
            print("  - traces.json")

        return results_dir


@register_command("run-episode")
def run_episode_command(args: argparse.Namespace) -> int:
    """Run a single episode with an LLM agent."""
    from salesbench.llm import get_api_key_for_provider
    from salesbench.models import DEFAULT_BUYER_MODEL, parse_model_spec

    # Parse seller model (CLI arg > env var)
    seller_model_str = args.seller_model or os.environ.get("SALESBENCH_SELLER_MODEL")
    if not seller_model_str:
        print("ERROR: No seller model specified.")
        print("Use --seller-model provider/model (e.g., --seller-model openai/gpt-4o)")
        return 1

    try:
        seller_spec = parse_model_spec(seller_model_str)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Parse buyer model (CLI arg > env var > default)
    buyer_model_str = (
        args.buyer_model or os.environ.get("SALESBENCH_BUYER_MODEL") or DEFAULT_BUYER_MODEL
    )
    try:
        buyer_spec = parse_model_spec(buyer_model_str)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Check API keys
    if not get_api_key_for_provider(seller_spec.provider):
        print(f"ERROR: No API key found for seller provider '{seller_spec.provider}'.")
        return 1

    if not get_api_key_for_provider(buyer_spec.provider):
        print(f"ERROR: No API key found for buyer provider '{buyer_spec.provider}'.")
        return 1

    # Run with telemetry context
    with telemetry_context(verbose=True) as telemetry_manager:
        runner = EpisodeRunner(
            args=args,
            seller_spec=seller_spec,
            buyer_spec=buyer_spec,
            telemetry_manager=telemetry_manager,
        )

        if not runner.setup():
            return 1

        # Run with or without tracing
        if runner.tracer:
            with runner.tracer.start_as_current_span("episode") as episode_span:
                turn_count = runner.run_episode_loop(episode_span)
                final = runner.orchestrator.get_final_result()
                # Add final metrics to episode span
                episode_span.set_attribute("episode.total_turns", turn_count)
                episode_span.set_attribute("episode.final_score", final.final_score)
                episode_span.set_attribute("episode.total_accepts", final.metrics.get("accepted_offers", 0))
                episode_span.set_attribute("episode.total_rejects", final.metrics.get("rejected_offers", 0))
                episode_span.set_attribute("episode.total_calls", final.metrics.get("total_calls", 0))
                episode_span.set_attribute("episode.dnc_violations", final.metrics.get("dnc_violations", 0))
                episode_span.add_event("episode_complete", {
                    "final_score": final.final_score,
                    "accepts": final.metrics.get("accepted_offers", 0),
                    "rejects": final.metrics.get("rejected_offers", 0),
                })
        else:
            runner.run_episode_loop(None)
            final = runner.orchestrator.get_final_result()

        runner.print_results(final)
        runner.save_results(final)

    return 0
