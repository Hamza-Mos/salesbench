"""SalesBench UI components.

Provides a Gradio-based leaderboard for visualizing benchmark results.
Compatible with HuggingFace Spaces deployment.
"""

from salesbench.ui.app import create_leaderboard

__all__ = ["create_leaderboard"]
