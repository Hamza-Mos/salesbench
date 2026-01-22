"""Leaderboard command for CLI.

Launches the Gradio leaderboard UI.
"""

import argparse

from salesbench.cli.commands import register_command


@register_command("leaderboard")
def leaderboard_command(args: argparse.Namespace) -> int:
    """Launch the leaderboard UI."""
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
