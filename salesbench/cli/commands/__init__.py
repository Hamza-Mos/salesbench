"""Command registry and dispatch for CLI.

Replaces elif chain with dictionary-based dispatch.
"""

import argparse
from typing import Callable, Dict

# Command function type
CommandFn = Callable[[argparse.Namespace], int]

# Command registry - populated by submodules
_COMMANDS: Dict[str, CommandFn] = {}


def register_command(name: str) -> Callable[[CommandFn], CommandFn]:
    """Decorator to register a command function.

    Args:
        name: Command name (e.g., 'seed-leads', 'run-episode').

    Returns:
        Decorator function.

    Example:
        @register_command("seed-leads")
        def seed_leads_command(args: argparse.Namespace) -> int:
            ...
    """
    def decorator(fn: CommandFn) -> CommandFn:
        _COMMANDS[name] = fn
        return fn
    return decorator


def get_command(name: str) -> CommandFn | None:
    """Get a registered command by name.

    Args:
        name: Command name.

    Returns:
        Command function or None if not found.
    """
    return _COMMANDS.get(name)


def dispatch_command(name: str, args: argparse.Namespace) -> int:
    """Dispatch to a registered command.

    Args:
        name: Command name.
        args: Parsed arguments.

    Returns:
        Exit code from command.

    Raises:
        KeyError: If command not found.
    """
    if name not in _COMMANDS:
        raise KeyError(f"Unknown command: {name}")
    return _COMMANDS[name](args)


def list_commands() -> list[str]:
    """List all registered command names."""
    return list(_COMMANDS.keys())


# Import submodules to trigger command registration
from salesbench.cli.commands import inspect  # noqa: E402, F401
from salesbench.cli.commands import leaderboard  # noqa: E402, F401
from salesbench.cli.commands import run_benchmark  # noqa: E402, F401
from salesbench.cli.commands import run_episode  # noqa: E402, F401
