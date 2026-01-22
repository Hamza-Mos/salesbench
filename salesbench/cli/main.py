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
import logging
import sys
from typing import Optional

from dotenv import load_dotenv

from salesbench.cli.arguments import (
    add_buyer_model_arg,
    add_domain_arg,
    add_format_arg,
    add_leads_arg,
    add_seed_arg,
    add_seller_model_arg,
    add_verbose_arg,
)
from salesbench.cli.commands import dispatch_command

# Load .env file from current directory
load_dotenv()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _setup_seed_leads_parser(subparsers) -> None:
    """Set up seed-leads command parser."""
    parser = subparsers.add_parser(
        "seed-leads",
        help="Generate and display personas from a seed",
    )
    add_seed_arg(parser)
    parser.add_argument("--n", "-n", type=int, default=100, help="Number of leads to generate (default: 100)")
    add_format_arg(parser)
    parser.add_argument("--full", action="store_true", help="Include hidden state in output")


def _setup_inspect_products_parser(subparsers) -> None:
    """Set up inspect-products command parser."""
    parser = subparsers.add_parser("inspect-products", help="Display product catalog")
    add_format_arg(parser)


def _setup_quote_parser(subparsers) -> None:
    """Set up quote command parser."""
    parser = subparsers.add_parser("quote", help="Get a premium quote")
    parser.add_argument("--plan", "-p", type=str, required=True, help="Plan type (TERM, WHOLE, UL, VUL, LTC, DI)")
    parser.add_argument("--age", "-a", type=int, required=True, help="Age of insured")
    parser.add_argument("--coverage", "-c", type=float, required=True, help="Coverage amount")
    parser.add_argument("--risk", "-r", type=str, default=None, help="Risk class (default: standard_plus)")
    parser.add_argument("--term", "-t", type=int, default=None, help="Term years (for TERM plans)")
    add_format_arg(parser)


def _setup_run_episode_parser(subparsers) -> None:
    """Set up run-episode command parser."""
    parser = subparsers.add_parser(
        "run-episode",
        help="Run a single episode with an LLM agent (requires OPENAI_API_KEY)",
    )
    add_seed_arg(parser)
    add_leads_arg(parser)
    parser.add_argument(
        "--max-turns", "-m", type=int, default=100,
        help="Safety turn limit (default: 100). Episodes end naturally via time limit or lead exhaustion.",
    )
    add_seller_model_arg(parser)
    add_buyer_model_arg(parser)
    add_verbose_arg(parser)


def _setup_run_benchmark_parser(subparsers) -> None:
    """Set up run-benchmark command parser."""
    parser = subparsers.add_parser(
        "run-benchmark",
        help="Run N episodes in parallel with Supabase/telemetry integrations",
    )
    parser.add_argument("--name", type=str, default="", help="Benchmark name/identifier (default: auto-generated)")
    parser.add_argument("--episodes", "-n", type=int, default=None, help="Number of episodes (default: based on mode)")
    add_seed_arg(parser)
    parser.add_argument("--leads", "-l", type=int, default=None, help="Leads per episode (default: based on mode)")
    parser.add_argument("--safety-max-turns", type=int, default=None, help="Safety turn limit per episode")
    parser.add_argument("--parallelism", "-p", type=int, default=None, help="Concurrent episodes (default: based on mode)")
    parser.add_argument("--progress-interval", type=int, default=5, help="Update progress display every N turns (default: 5)")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Seller model(s) to benchmark. Format: provider/model. Examples: 'openai/gpt-4o' or 'openai/gpt-4o,anthropic/claude-sonnet-4-20250514'",
    )
    add_buyer_model_arg(parser)
    parser.add_argument(
        "--mode", type=str, choices=["production", "demo", "test", "debug"], default="production",
        help="Run mode: production (100 eps), demo (10 eps), test (3 eps), debug (1 ep)",
    )
    parser.add_argument("--no-supabase", action="store_true", help="Disable Supabase storage")
    parser.add_argument("--telemetry", action="store_true", help="Enable OpenTelemetry")
    parser.add_argument("--output", "-o", type=str, default=None, help="Custom name for results directory")
    add_verbose_arg(parser)
    add_domain_arg(parser)
    parser.add_argument("--hours", type=int, default=None, help="Total simulated hours (override preset)")


def _setup_list_models_parser(subparsers) -> None:
    """Set up list-models command parser."""
    parser = subparsers.add_parser("list-models", help="List known models and providers")
    add_format_arg(parser)


def _setup_list_domains_parser(subparsers) -> None:
    """Set up list-domains command parser."""
    parser = subparsers.add_parser("list-domains", help="List available sales domains")
    add_format_arg(parser)


def _setup_leaderboard_parser(subparsers) -> None:
    """Set up leaderboard command parser."""
    parser = subparsers.add_parser("leaderboard", help="Launch the leaderboard UI (requires gradio)")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory containing result files (default: results/)")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public share link")


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="salesbench",
        description="SalesBench - AI Social Intelligence Benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    _setup_seed_leads_parser(subparsers)
    _setup_inspect_products_parser(subparsers)
    _setup_quote_parser(subparsers)
    _setup_run_episode_parser(subparsers)
    _setup_run_benchmark_parser(subparsers)
    _setup_list_models_parser(subparsers)
    _setup_list_domains_parser(subparsers)
    _setup_leaderboard_parser(subparsers)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Configure logging for commands that have verbose flag
    if args.command in ("run-episode", "run-benchmark") and hasattr(args, "verbose"):
        setup_logging(args.verbose)

    # Dispatch to registered command
    try:
        return dispatch_command(args.command, args)
    except KeyError:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
