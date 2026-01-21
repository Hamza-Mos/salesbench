"""Minimal internal stub of the `verifiers` package used by SalesBench.

SalesBench includes a verifiers-compatible environment in `salesbench/environment.py`.
The upstream `verifiers` dependency is optional, but our unit tests expect that
`import verifiers as vf` works in this repo.

This module provides just enough surface area for SalesBench and its tests:
- `StatefulToolEnv`
- `Rubric`
- `State` / `Messages` aliases
- `stop` decorator

It is intentionally minimal and not a full implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

State = dict[str, Any]
Messages = list[dict[str, Any]]


def stop(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator stub used by verifiers to mark stopping conditions."""
    return func


@dataclass
class Rubric:
    """Minimal rubric container."""

    funcs: list[Callable[..., Any]]
    weights: list[float]


class StatefulToolEnv:
    """Minimal base class to satisfy SalesBenchToolEnv usage in tests."""

    def __init__(
        self,
        *,
        dataset: Any,
        tools: list[Callable[..., Any]],
        max_turns: int = 100,
        rubric: Optional[Rubric] = None,
        system_prompt: str = "",
        **kwargs: Any,
    ):
        self.dataset = dataset
        self.tools: list[Callable[..., Any]] = list(tools)
        self.max_turns = max_turns
        self.rubric = rubric
        self.system_prompt = system_prompt

    def add_tool(
        self, tool_func: Callable[..., Any], args_to_skip: Optional[list[str]] = None
    ) -> None:
        """Register a tool. `args_to_skip` is accepted for compatibility only."""
        self.tools.append(tool_func)
