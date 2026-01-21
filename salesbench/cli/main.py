"""CLI entrypoint for SalesBench.

Commands:
- run-benchmark: Run benchmark with one or more models
- run-episode: Run a single episode (for debugging)
- list-models: List known models and providers
- list-domains: List available sales domains
- seed-leads: Generate and display personas from a seed
- inspect-products: Display product catalog
- leaderboard: Launch the leaderboard UI
"""

import argparse
import json
import os
import sys
from typing import Optional

from dotenv import load_dotenv

# Load .env file from current directory
load_dotenv()


def seed_leads_command(args: argparse.Namespace) -> int:
    """Generate and display personas from a seed.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    from salesbench.envs.sales_mvp.personas import PersonaGenerator

    generator = PersonaGenerator(seed=args.seed)
    leads = generator.generate_batch(args.n)

    if args.format == "json":
        output = {
            "seed": args.seed,
            "count": len(leads),
            "leads": [lead.to_public_dict() for lead in leads],
        }
        if args.full:
            output["leads"] = [lead.to_full_dict() for lead in leads]
        print(json.dumps(output, indent=2))
    else:
        # Table format
        print(f"Generated {len(leads)} leads with seed {args.seed}")
        print("-" * 80)

        # Count by temperature
        temps = {}
        for lead in leads:
            t = lead.temperature.value
            temps[t] = temps.get(t, 0) + 1

        print("\nTemperature distribution:")
        for temp, count in sorted(temps.items()):
            pct = count / len(leads) * 100
            print(f"  {temp:10s}: {count:3d} ({pct:5.1f}%)")

        print("\nSample leads:")
        print(f"{'Name':<25} {'Age':>4} {'Temperature':<10} {'Income':>10} {'Job':<25}")
        print("-" * 80)
        for lead in leads[:10]:
            print(
                f"{lead.name:<25} {lead.age:>4} {lead.temperature.value:<10} "
                f"${lead.annual_income:>9,} {lead.job:<25}"
            )

        if args.full:
            print("\nHidden state (first 5 leads):")
            for lead in leads[:5]:
                print(f"  {lead.name}:")
                print(f"    trust={lead.hidden.trust:.2f}, interest={lead.hidden.interest:.2f}")
                print(
                    f"    patience={lead.hidden.patience:.2f}, dnc_risk={lead.hidden.dnc_risk:.2f}"
                )
                print(f"    close_threshold={lead.hidden.close_threshold:.2%}")

    return 0


def inspect_products_command(args: argparse.Namespace) -> int:
    """Display product catalog.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    from salesbench.envs.sales_mvp.products import ProductCatalog

    catalog = ProductCatalog()
    products = catalog.list_products()

    if args.format == "json":
        print(json.dumps(products, indent=2))
    else:
        print("Insurance Product Catalog")
        print("=" * 80)
        for product in products:
            print(f"\n{product['name']} ({product['plan_id']})")
            print("-" * 40)
            print(f"  {product['description'][:70]}...")
            print(f"  Coverage: ${product['min_coverage']:,} - ${product['max_coverage']:,}")
            print(f"  Ages: {product['min_age']} - {product['max_age']}")
            print("  Features:")
            for feature in product["features"][:3]:
                print(f"    - {feature}")

    return 0


def quote_command(args: argparse.Namespace) -> int:
    """Get a premium quote.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    from salesbench.core.types import PlanType, RiskClass
    from salesbench.envs.sales_mvp.products import ProductCatalog

    catalog = ProductCatalog()

    try:
        plan_type = PlanType(args.plan)
    except ValueError:
        print(f"Invalid plan: {args.plan}")
        print(f"Valid plans: {[p.value for p in PlanType]}")
        return 1

    risk_class = RiskClass.STANDARD_PLUS
    if args.risk:
        try:
            risk_class = RiskClass(args.risk)
        except ValueError:
            print(f"Invalid risk class: {args.risk}")
            print(f"Valid: {[r.value for r in RiskClass]}")
            return 1

    quote = catalog.quote_premium(
        plan_id=plan_type,
        age=args.age,
        coverage_amount=args.coverage,
        risk_class=risk_class,
        term_years=args.term,
    )

    if "error" in quote:
        print(f"Error: {quote['error']}")
        return 1

    if args.format == "json":
        print(json.dumps(quote, indent=2))
    else:
        print("\nPremium Quote")
        print("=" * 40)
        print(f"  Plan: {quote['plan_name']}")
        print(f"  Age: {quote['age']}")
        print(f"  Risk Class: {quote['risk_class']}")
        if "term_years" in quote:
            print(f"  Term: {quote['term_years']} years")
        print(f"  Coverage: ${quote.get('coverage_amount', 0):,.0f}")
        print("-" * 40)
        print(f"  Monthly Premium: ${quote['monthly_premium']:,.2f}")
        print(f"  Annual Premium: ${quote['annual_premium']:,.2f}")
        if "projected_cash_value_year_10" in quote:
            print(
                f"  Projected Cash Value (Year 10): ${quote['projected_cash_value_year_10']:,.2f}"
            )

    return 0


def run_episode_command(args: argparse.Namespace) -> int:
    """Run a single episode with an LLM agent.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    # Initialize telemetry if enabled
    telemetry_manager = None
    try:
        from salesbench.telemetry.otel import TelemetryConfig, TelemetryManager

        telemetry_config = TelemetryConfig()
        if telemetry_config.enabled:
            telemetry_manager = TelemetryManager(telemetry_config)
            if telemetry_manager.init():
                print("Telemetry: enabled (traces will be sent to Grafana)")
            else:
                print("Telemetry: failed to initialize")
                telemetry_manager = None
        else:
            print("Telemetry: disabled (set OTEL_ENABLED=true to enable)")
    except ImportError:
        print("Telemetry: not available (missing opentelemetry packages)")

    from salesbench.agents.buyer_llm import create_buyer_simulator
    from salesbench.agents.seller_llm import LLMSellerAgent
    from salesbench.core.config import SalesBenchConfig
    from salesbench.llm import get_api_key_for_provider
    from salesbench.models import DEFAULT_BUYER_MODEL, parse_model_spec
    from salesbench.orchestrator.orchestrator import Orchestrator

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

    config = SalesBenchConfig(seed=args.seed, num_leads=args.leads)
    orchestrator = Orchestrator(config)

    # Set up LLM buyer simulator with explicit provider
    buyer_temp = float(os.environ.get("SALESBENCH_BUYER_TEMPERATURE", "0.0"))
    buyer_simulator = create_buyer_simulator(
        provider=buyer_spec.provider,
        model=buyer_spec.model,
        temperature=buyer_temp,
    )
    orchestrator.set_buyer_simulator(buyer_simulator)

    # Set up LLM seller agent with explicit provider
    seller_agent = LLMSellerAgent(
        provider=seller_spec.provider,
        model=seller_spec.model,
    )

    print(f"Starting episode with seed {args.seed}, {args.leads} leads")
    print(f"Seller: {seller_spec}")
    print(f"Buyer: {buyer_spec}")
    print("-" * 60)

    # Get telemetry span manager if available
    span_manager = None
    if telemetry_manager:
        try:
            from salesbench.telemetry.spans import get_span_manager
            span_manager = get_span_manager()
        except ImportError:
            pass

    from salesbench.agents.seller_base import SellerObservation
    from salesbench.core.types import ToolResult

    def dict_to_observation(obs_dict: dict) -> SellerObservation:
        """Convert orchestrator dict to SellerObservation."""
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

    obs_dict = orchestrator.reset()
    print(f"Time: Day {obs_dict['time']['current_day']}, {obs_dict['time']['current_hour']}:00")
    print(f"Leads available: {obs_dict['leads_count']}")

    # Get tool schemas for the agent
    tool_schemas = orchestrator.env.get_tools_schema()

    # Create episode span for telemetry
    import uuid
    from contextlib import nullcontext
    episode_id = f"episode_{uuid.uuid4().hex[:8]}"

    # Get tracer for detailed spans
    tracer = None
    if telemetry_manager:
        try:
            from salesbench.telemetry.otel import get_tracer
            tracer = get_tracer()
        except ImportError:
            pass

    # Display names for telemetry
    display_seller_model = str(seller_spec)
    display_buyer_model = str(buyer_spec)

    def run_episode_with_tracing(episode_span):
        """Run episode loop, logging to the provided span."""
        nonlocal obs_dict
        turn = 0

        # Set episode attributes
        if episode_span:
            episode_span.set_attribute("episode.id", episode_id)
            episode_span.set_attribute("episode.seed", args.seed)
            episode_span.set_attribute("episode.num_leads", args.leads)
            episode_span.set_attribute("model.seller", display_seller_model)
            episode_span.set_attribute("model.buyer", display_buyer_model)

        while not orchestrator.is_terminated and turn < args.max_turns:
            turn += 1

            # Create turn span as child of episode (using context manager for proper nesting)
            turn_context = tracer.start_as_current_span(f"turn_{turn}") if tracer else nullcontext()

            with turn_context as turn_span:
                if turn_span:
                    turn_span.set_attribute("turn.number", turn)

                # Convert dict to SellerObservation for the agent
                obs = dict_to_observation(obs_dict)

                # Build the prompt that will be sent to LLM (for logging)
                # This mirrors what _build_user_message does in the agent
                prompt_summary = f"Day {obs.current_day}, Hour {obs.current_hour}, "
                prompt_summary += f"Calls: {obs.total_calls}, Accepts: {obs.total_accepts}"
                if obs.in_call:
                    prompt_summary += f" | IN CALL with {obs.current_lead_id}"

                # LLM agent generates action (message + tool calls)
                try:
                    # LLM call span nested under turn span
                    llm_context = tracer.start_as_current_span("llm_call") if tracer else nullcontext()
                    with llm_context as llm_span:
                        if llm_span:
                            llm_span.set_attribute("llm.model", display_seller_model)
                            # Log the input context
                            llm_span.add_event("llm_input", {
                                "prompt_summary": prompt_summary,
                                "in_call": obs.in_call,
                                "current_lead": obs.current_lead_id or "",
                                "day": obs.current_day,
                                "hour": obs.current_hour,
                            })

                        action = seller_agent.act(
                            obs,
                            tool_schemas,
                            episode_context=orchestrator.episode_context,
                        )

                        if llm_span and action:
                            llm_span.set_attribute("llm.has_message", bool(action.message))
                            llm_span.set_attribute("llm.tool_count", len(action.tool_calls))

                            # Log the full LLM response
                            response_text = action.message or ""
                            tool_calls_str = ", ".join([
                                f"{tc.tool_name}({json.dumps(tc.arguments)})"
                                for tc in action.tool_calls
                            ]) if action.tool_calls else ""

                            # Add response as event with full text
                            llm_span.add_event("llm_output", {
                                "response_message": response_text[:2000] if response_text else "",
                                "tool_calls": tool_calls_str[:1000] if tool_calls_str else "",
                            })

                            # Also store full response as attribute (for easy viewing)
                            if response_text:
                                llm_span.set_attribute("llm.response", response_text[:4000])
                            if tool_calls_str:
                                llm_span.set_attribute("llm.tool_calls", tool_calls_str[:2000])

                except Exception as e:
                    print(f"\nError from LLM seller: {e}")
                    if turn_span:
                        turn_span.set_attribute("turn.error", str(e))
                    break

                # Extract tool calls from action
                tool_calls = action.tool_calls if action else []
                message = action.message if action else None

                # Log message to span (full text)
                if turn_span and message:
                    turn_span.set_attribute("seller.message", message[:2000] if len(message) > 2000 else message)
                    turn_span.add_event("seller_message", {"message": message[:500]})

                if not tool_calls and not message:
                    print(f"\nTurn {turn}: Agent returned no action, ending episode.")
                    if turn_span:
                        turn_span.add_event("no_action", {"reason": "agent returned empty"})
                    break

                # Log tool calls to span
                if turn_span and tool_calls:
                    tool_names = [tc.tool_name for tc in tool_calls]
                    turn_span.set_attribute("tools.called", ",".join(tool_names))

                # Step environment with tool calls
                result = orchestrator.step(tool_calls) if tool_calls else None

                # Process results and log to span
                if result:
                    for tr in result.tool_results:
                        if turn_span:
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
                                # Log full buyer dialogue
                                if "dialogue" in tr.data and tr.data["dialogue"]:
                                    buyer_text = tr.data["dialogue"]
                                    event_attrs["buyer_dialogue"] = buyer_text[:500]
                                    # Also add as separate attribute for easy viewing
                                    turn_span.set_attribute("buyer.response", buyer_text[:2000])
                                # Log seller pitch if present
                                if "seller_pitch" in tr.data and tr.data["seller_pitch"]:
                                    event_attrs["seller_pitch"] = tr.data["seller_pitch"][:500]
                            turn_span.add_event("tool_result", event_attrs)

                if args.verbose:
                    print(f"\nTurn {turn}:")
                    if message:
                        print(f"  💬 {message[:200]}{'...' if len(message) > 200 else ''}")
                    for tc in tool_calls:
                        print(f"  → {tc.tool_name}({tc.arguments})")
                    if result:
                        for tr in result.tool_results:
                            status = "OK" if tr.success else "FAIL"
                            print(f"  ← [{status}] {tr.data or tr.error}")
                            if tr.success and tr.data:
                                if "seller_pitch" in tr.data and tr.data["seller_pitch"]:
                                    print(f"  🗣️  Seller: \"{tr.data['seller_pitch']}\"")
                                if "dialogue" in tr.data and tr.data["dialogue"]:
                                    print(f"  📞 Buyer: \"{tr.data['dialogue']}\"")

                if result and result.terminated:
                    print(f"\nEpisode terminated: {result.termination_reason}")
                    if turn_span:
                        turn_span.add_event("episode_terminated", {"reason": result.termination_reason})
                    break

                if result:
                    obs_dict = result.observation
                    obs_dict["last_tool_results"] = [tr.to_dict() for tr in result.tool_results]

        # Return turn count for final metrics
        return turn

    # Run with episode span as the root (all child spans will be nested under it)
    if tracer:
        with tracer.start_as_current_span("episode") as episode_span:
            turn_count = run_episode_with_tracing(episode_span)
            final = orchestrator.get_final_result()
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
        run_episode_with_tracing(None)
        final = orchestrator.get_final_result()
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
    seller_in, seller_out = seller_agent.get_token_usage()
    buyer_in, buyer_out = 0, 0
    print(f"  Seller: {seller_in:,} input, {seller_out:,} output")
    if hasattr(buyer_simulator, "get_token_usage"):
        buyer_in, buyer_out = buyer_simulator.get_token_usage()
        print(f"  Buyer:  {buyer_in:,} input, {buyer_out:,} output")
    print(f"  Total:  {seller_in + buyer_in:,} input, {seller_out + buyer_out:,} output")

    # Save results to timestamped directory
    from salesbench.runner.results import EpisodeResult, TokenUsage
    from salesbench.storage.json_writer import JSONResultsWriter

    # Build episode result
    token_usage = TokenUsage()
    token_usage.add_seller_usage(seller_in, seller_out)
    token_usage.add_buyer_usage(buyer_in, buyer_out)

    episode_result = {
        "mode": "single_episode",
        "config": {
            "seed": args.seed,
            "num_leads": args.leads,
            "max_turns": args.max_turns,
            "seller_model": str(seller_spec),
            "buyer_model": str(buyer_spec),
        },
        "episode_results": [
            {
                "episode_id": f"ep_{args.seed}",
                "episode_index": 0,
                "seed": args.seed,
                "status": "completed",
                "final_score": final.final_score,
                "total_turns": final.total_turns,
                "total_accepts": final.metrics.get("accepted_offers", 0),
                "total_rejects": final.metrics.get("rejected_offers", 0),
                "total_calls": final.metrics.get("total_calls", 0),
                "dnc_violations": final.metrics.get("dnc_violations", 0),
                "metrics": final.metrics,
                "trajectory": final.history if args.verbose else [],
                "token_usage": token_usage.to_dict(),
            }
        ],
        "aggregate_metrics": {
            "total_token_usage": token_usage.to_dict(),
        },
    }

    # Extract model name for directory
    model_name = seller_spec.model

    writer = JSONResultsWriter("results")
    results_dir = writer.write_benchmark(
        benchmark_id=f"episode_{args.seed}",
        result=episode_result,
        include_traces=args.verbose,
        custom_name=f"episode_{model_name}",
    )
    print(f"\nResults saved to: {results_dir}/")
    print("  - summary.json")
    if args.verbose:
        print("  - traces.json")

    # Shutdown telemetry (flushes traces)
    if telemetry_manager:
        print("\nFlushing telemetry traces...")
        telemetry_manager.shutdown()
        print("Telemetry: traces sent to Grafana")

    return 0


def run_benchmark_command(args: argparse.Namespace) -> int:
    """Run a benchmark with multiple episodes.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    from salesbench.models import (
        DEFAULT_BUYER_MODEL,
        get_default_benchmark_models,
        parse_model_list,
        parse_model_spec,
    )
    from salesbench.runner import BenchmarkConfig, BenchmarkRunner

    # Determine which models to benchmark
    if args.models:
        models = parse_model_list(args.models)
    else:
        # No models specified - use default benchmark set
        models = get_default_benchmark_models()
        print("No --models specified. Using default set:")
        for m in models:
            print(f"  - {m}")

    # Parse buyer model
    buyer_model = (
        args.buyer_model or os.environ.get("SALESBENCH_BUYER_MODEL") or DEFAULT_BUYER_MODEL
    )
    buyer_spec = parse_model_spec(buyer_model)

    # Run benchmarks for all models
    print(f"\nRunning benchmark for {len(models)} model(s)")
    print(f"Buyer model: {buyer_spec}")
    print("=" * 60)

    all_passed = True
    for i, model_spec in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Benchmarking: {model_spec}")
        print("-" * 40)

        config = BenchmarkConfig.from_cli_args(
            mode=args.mode,
            episodes=args.episodes,
            seed=args.seed,
            leads=args.leads,
            max_turns=args.max_turns,
            parallelism=args.parallelism,
            seller_model=str(model_spec),
            buyer_model=str(buyer_spec),
            no_supabase=args.no_supabase,
            enable_telemetry=args.telemetry,
            output=args.output,
            verbose=args.verbose,
            name=args.name or "",
            domain=args.domain,
        )

        runner = BenchmarkRunner(config)
        result = runner.run()

        if result.failed_episodes > 0:
            all_passed = False

    print("\n" + "=" * 60)
    print(f"Completed benchmarks for {len(models)} model(s)")

    return 0 if all_passed else 1


def list_models_command(args: argparse.Namespace) -> int:
    """List known models and providers.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    from salesbench.models import DEFAULT_BENCHMARK_MODELS, list_known_models

    models_by_provider = list_known_models()

    if args.format == "json":
        output = {
            "models_by_provider": models_by_provider,
            "default_benchmark_set": DEFAULT_BENCHMARK_MODELS,
        }
        print(json.dumps(output, indent=2))
    else:
        print("Known Models by Provider")
        print("=" * 60)
        for provider, models in sorted(models_by_provider.items()):
            print(f"\n{provider.upper()}")
            print("-" * 40)
            for model in models:
                marker = " *" if model in DEFAULT_BENCHMARK_MODELS else ""
                print(f"  {model}{marker}")

        print("\n" + "=" * 60)
        print("Default benchmark set (* above):")
        for model in DEFAULT_BENCHMARK_MODELS:
            print(f"  - {model}")

        print("\nUsage:")
        print("  salesbench run-benchmark                           # Runs default set")
        print("  salesbench run-benchmark --models openai/gpt-4o    # Single model")
        print(
            "  salesbench run-benchmark --models openai/gpt-4o,anthropic/claude-sonnet-4-20250514"
        )

    return 0


def list_domains_command(args: argparse.Namespace) -> int:
    """List available sales domains.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    # Import domains package to trigger registration
    import salesbench.domains.insurance  # noqa: F401
    from salesbench.domains import get_domain, list_domains

    domains = list_domains()

    if args.format == "json":
        output = []
        for name in domains:
            domain = get_domain(name)
            output.append(
                {
                    "name": domain.config.name,
                    "display_name": domain.config.display_name,
                    "description": domain.config.description,
                    "product_types": domain.config.product_types,
                    "tools": domain.config.tools,
                }
            )
        print(json.dumps(output, indent=2))
    else:
        print("Available Sales Domains")
        print("=" * 60)
        for name in domains:
            domain = get_domain(name)
            print(f"\n{domain.config.display_name} [{name}]")
            print(f"  {domain.config.description}")
            print(f"  Products: {', '.join(domain.config.product_types)}")
            print(f"  Tools: {len(domain.config.tools)} available")

    return 0


def leaderboard_command(args: argparse.Namespace) -> int:
    """Launch the leaderboard UI.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    try:
        from salesbench.ui.app import create_leaderboard
    except ImportError:
        print("ERROR: Gradio is required for the leaderboard UI.")
        print("Install with: pip install 'salesbench[ui]'")
        return 1

    demo = create_leaderboard(results_dir=args.results_dir)
    demo.launch(
        server_port=args.port,
        share=args.share,
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command line arguments (defaults to sys.argv).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="salesbench",
        description="SalesBench - AI Social Intelligence Benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # seed-leads command
    seed_parser = subparsers.add_parser(
        "seed-leads",
        help="Generate and display personas from a seed",
    )
    seed_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    seed_parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=100,
        help="Number of leads to generate (default: 100)",
    )
    seed_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    seed_parser.add_argument(
        "--full",
        action="store_true",
        help="Include hidden state in output",
    )

    # inspect-products command
    products_parser = subparsers.add_parser(
        "inspect-products",
        help="Display product catalog",
    )
    products_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # quote command
    quote_parser = subparsers.add_parser(
        "quote",
        help="Get a premium quote",
    )
    quote_parser.add_argument(
        "--plan",
        "-p",
        type=str,
        required=True,
        help="Plan type (TERM, WHOLE, UL, VUL, LTC, DI)",
    )
    quote_parser.add_argument(
        "--age",
        "-a",
        type=int,
        required=True,
        help="Age of insured",
    )
    quote_parser.add_argument(
        "--coverage",
        "-c",
        type=float,
        required=True,
        help="Coverage amount",
    )
    quote_parser.add_argument(
        "--risk",
        "-r",
        type=str,
        default=None,
        help="Risk class (default: standard_plus)",
    )
    quote_parser.add_argument(
        "--term",
        "-t",
        type=int,
        default=None,
        help="Term years (for TERM plans)",
    )
    quote_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # run-episode command
    run_parser = subparsers.add_parser(
        "run-episode",
        help="Run a single episode with an LLM agent (requires OPENAI_API_KEY)",
    )
    run_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    run_parser.add_argument(
        "--leads",
        "-l",
        type=int,
        default=20,
        help="Number of leads (default: 20)",
    )
    run_parser.add_argument(
        "--max-turns",
        "-m",
        type=int,
        default=50,
        help="Maximum turns (default: 50)",
    )
    run_parser.add_argument(
        "--seller-model",
        type=str,
        default=None,
        help="Seller model. Format: provider/model (e.g., openai/gpt-4o)",
    )
    run_parser.add_argument(
        "--buyer-model",
        type=str,
        default=None,
        help="Buyer model (default: openai/gpt-4o-mini). Format: provider/model",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    # run-benchmark command
    benchmark_parser = subparsers.add_parser(
        "run-benchmark",
        help="Run N episodes in parallel with Supabase/telemetry integrations",
    )
    benchmark_parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Benchmark name/identifier (default: auto-generated)",
    )
    benchmark_parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=None,
        help="Number of episodes (default: based on mode)",
    )
    benchmark_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    benchmark_parser.add_argument(
        "--leads",
        "-l",
        type=int,
        default=None,
        help="Leads per episode (default: based on mode)",
    )
    benchmark_parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum turns per episode (default: based on mode)",
    )
    benchmark_parser.add_argument(
        "--parallelism",
        "-p",
        type=int,
        default=None,
        help="Concurrent episodes (default: based on mode)",
    )
    benchmark_parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Seller model(s) to benchmark. Format: provider/model. "
        "Examples: 'openai/gpt-4o' or 'openai/gpt-4o,anthropic/claude-sonnet-4-20250514'. "
        "If omitted, benchmarks default model set.",
    )
    benchmark_parser.add_argument(
        "--buyer-model",
        type=str,
        default=None,
        help="Buyer simulator model (default: openai/gpt-4o-mini). Format: provider/model.",
    )
    benchmark_parser.add_argument(
        "--mode",
        type=str,
        choices=["production", "test", "debug"],
        default="production",
        help="Run mode: production (100 episodes), test (3 episodes), debug (1 episode)",
    )
    benchmark_parser.add_argument(
        "--no-supabase",
        action="store_true",
        help="Disable Supabase storage",
    )
    benchmark_parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Enable OpenTelemetry (requires OTEL collector at localhost:4317)",
    )
    benchmark_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom name for results directory (default: model name)",
    )
    benchmark_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output + save conversation traces to traces.json",
    )
    benchmark_parser.add_argument(
        "--domain",
        "-d",
        type=str,
        default="insurance",
        help="Sales domain to benchmark (default: insurance)",
    )

    # list-models command
    models_parser = subparsers.add_parser(
        "list-models",
        help="List known models and providers",
    )
    models_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # list-domains command
    domains_parser = subparsers.add_parser(
        "list-domains",
        help="List available sales domains",
    )
    domains_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # leaderboard command
    leaderboard_parser = subparsers.add_parser(
        "leaderboard",
        help="Launch the leaderboard UI (requires gradio)",
    )
    leaderboard_parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing result files (default: results/)",
    )
    leaderboard_parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    leaderboard_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "seed-leads":
        return seed_leads_command(args)
    elif args.command == "inspect-products":
        return inspect_products_command(args)
    elif args.command == "quote":
        return quote_command(args)
    elif args.command == "run-episode":
        return run_episode_command(args)
    elif args.command == "run-benchmark":
        return run_benchmark_command(args)
    elif args.command == "list-models":
        return list_models_command(args)
    elif args.command == "list-domains":
        return list_domains_command(args)
    elif args.command == "leaderboard":
        return leaderboard_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
