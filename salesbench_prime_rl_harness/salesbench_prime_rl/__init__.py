"""SalesBench Prime RL package."""

from __future__ import annotations

from typing import Any

import verifiers as vf


def load_environment(*args: Any, **kwargs: Any) -> vf.Environment:
    from .environment import load_environment as _load_environment

    return _load_environment(*args, **kwargs)


def __getattr__(name: str):
    if name == "SalesBenchPrimeRLEnv":
        from .environment import SalesBenchPrimeRLEnv

        return SalesBenchPrimeRLEnv
    raise AttributeError(f"module 'salesbench_prime_rl' has no attribute {name!r}")


__all__ = ["load_environment", "SalesBenchPrimeRLEnv"]
