"""Shared argument builders for CLI commands.

Consolidates repeated argument definitions to eliminate duplication.
"""

import argparse


def add_format_arg(parser: argparse.ArgumentParser) -> None:
    """Add --format argument for output format selection."""
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )


def add_seed_arg(parser: argparse.ArgumentParser, default: int = 42) -> None:
    """Add --seed argument for random seed."""
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=default,
        help=f"Random seed (default: {default})",
    )


def add_leads_arg(parser: argparse.ArgumentParser, default: int = 20) -> None:
    """Add --leads argument for number of leads."""
    parser.add_argument(
        "--leads",
        "-l",
        type=int,
        default=default,
        help=f"Number of leads (default: {default})",
    )


def add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    """Add --verbose argument for verbose output."""
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )


def add_seller_model_arg(parser: argparse.ArgumentParser) -> None:
    """Add --seller-model argument."""
    parser.add_argument(
        "--seller-model",
        type=str,
        default=None,
        help="Seller model. Format: provider/model (e.g., openai/gpt-4o)",
    )


def add_buyer_model_arg(parser: argparse.ArgumentParser) -> None:
    """Add --buyer-model argument."""
    parser.add_argument(
        "--buyer-model",
        type=str,
        default=None,
        help="Buyer model (default: openai/gpt-4o-mini). Format: provider/model",
    )


def add_domain_arg(parser: argparse.ArgumentParser, default: str = "insurance") -> None:
    """Add --domain argument for sales domain."""
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        default=default,
        help=f"Sales domain to benchmark (default: {default})",
    )
