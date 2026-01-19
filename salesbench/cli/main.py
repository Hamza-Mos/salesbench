"""CLI entrypoint for SalesBench debug tools.

Commands:
- seed-leads: Generate and display personas from a seed
- run-episode: Run a single episode with an LLM agent
- run-benchmark: Run N episodes in parallel with integrations
- inspect-products: Display product catalog
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
    from salesbench.llm import detect_available_provider

    # Check for any LLM provider
    provider = detect_available_provider()
    if not provider:
        print("ERROR: No LLM provider API key found.")
        print("Set ONE of these environment variables in .env or shell:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - OPENROUTER_API_KEY")
        print("  - XAI_API_KEY")
        print("  - TOGETHER_API_KEY")
        print("  - GOOGLE_API_KEY")
        return 1

    from salesbench.agents.buyer_llm import create_buyer_simulator
    from salesbench.agents.seller_llm import LLMSellerAgent
    from salesbench.core.config import SalesBenchConfig
    from salesbench.orchestrator.orchestrator import Orchestrator

    config = SalesBenchConfig(seed=args.seed, num_leads=args.leads)
    orchestrator = Orchestrator(config)

    # Set up LLM buyer simulator (CLI arg > env var > default)
    buyer_model = args.buyer_model or os.environ.get("SALESBENCH_BUYER_MODEL")
    buyer_temp = float(os.environ.get("SALESBENCH_BUYER_TEMPERATURE", "0.3"))
    buyer_simulator = create_buyer_simulator(
        provider=provider,
        model=buyer_model,
        temperature=buyer_temp,
    )
    orchestrator.set_buyer_simulator(buyer_simulator)

    # Set up LLM seller agent (CLI arg > env var > default)
    seller_model = args.seller_model or os.environ.get("SALESBENCH_SELLER_MODEL")
    seller_agent = LLMSellerAgent(
        provider=provider,
        model=seller_model,
    )

    # Get actual model names for display
    from salesbench.llm import DEFAULT_MODELS

    display_buyer_model = buyer_model or DEFAULT_MODELS.get(provider, "default")
    display_seller_model = seller_model or DEFAULT_MODELS.get(provider, "default")

    print(f"Starting episode with seed {args.seed}, {args.leads} leads")
    print(f"Provider: {provider}")
    print(f"Buyer model: {display_buyer_model}, Seller model: {display_seller_model}")
    print("-" * 60)

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

    turn = 0
    while not orchestrator.is_terminated and turn < args.max_turns:
        turn += 1

        # Convert dict to SellerObservation for the agent
        obs = dict_to_observation(obs_dict)

        # LLM agent generates action (message + tool calls)
        try:
            action = seller_agent.act(obs, tool_schemas)
        except Exception as e:
            print(f"\nError from LLM seller: {e}")
            break

        # Extract tool calls from action
        tool_calls = action.tool_calls if action else []
        message = action.message if action else None

        if not tool_calls and not message:
            print(f"\nTurn {turn}: Agent returned no action, ending episode.")
            break

        # Step environment with tool calls (message is handled separately)
        result = orchestrator.step(tool_calls) if tool_calls else None

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
                    # Show conversation after propose_plan
                    if tr.success and tr.data:
                        # Show seller's pitch if provided
                        if "seller_pitch" in tr.data and tr.data["seller_pitch"]:
                            print(f"  🗣️  Seller: \"{tr.data['seller_pitch']}\"")
                        # Show buyer's dialogue
                        if "dialogue" in tr.data and tr.data["dialogue"]:
                            print(f"  📞 Buyer: \"{tr.data['dialogue']}\"")

        if result and result.terminated:
            print(f"\nEpisode terminated: {result.termination_reason}")
            break

        if result:
            obs_dict = result.observation
            # CRITICAL: Include tool results in the observation for the agent
            obs_dict["last_tool_results"] = [tr.to_dict() for tr in result.tool_results]

    # Print final results
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

    return 0


def run_benchmark_command(args: argparse.Namespace) -> int:
    """Run a benchmark with multiple episodes.

    Args:
        args: Command line arguments.

    Returns:
        Exit code.
    """
    from salesbench.runner import BenchmarkConfig, BenchmarkRunner

    # Build config from CLI args
    config = BenchmarkConfig.from_cli_args(
        mode=args.mode,
        episodes=args.episodes,
        seed=args.seed,
        leads=args.leads,
        max_turns=args.max_turns,
        parallelism=args.parallelism,
        seller_model=args.seller_model,
        buyer_model=args.buyer_model,
        no_supabase=args.no_supabase,
        no_telemetry=args.no_telemetry,
        output=args.output,
        verbose=args.verbose,
        name=args.name or "",
    )

    # Run benchmark
    runner = BenchmarkRunner(config)
    result = runner.run()

    # Return non-zero if any episodes failed
    if result.failed_episodes > 0:
        return 1
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
        help="Seller agent model (default: from env or gpt-4o-mini)",
    )
    run_parser.add_argument(
        "--buyer-model",
        type=str,
        default=None,
        help="Buyer agent model (default: from env or gpt-4o-mini)",
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
        "--seller-model",
        type=str,
        default=None,
        help="Seller agent model (default: from env)",
    )
    benchmark_parser.add_argument(
        "--buyer-model",
        type=str,
        default=None,
        help="Buyer simulator model (default: from env)",
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
        "--no-telemetry",
        action="store_true",
        help="Disable OpenTelemetry/Grafana",
    )
    benchmark_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="JSON output file path",
    )
    benchmark_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
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
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
