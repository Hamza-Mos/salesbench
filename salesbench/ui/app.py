"""Gradio leaderboard for SalesBench.

A simple, HuggingFace Spaces-compatible leaderboard that displays
benchmark results from JSON files.

Usage:
    python -m salesbench.ui.app

Or deploy to HuggingFace Spaces by pushing this file along with
requirements.txt containing 'gradio>=4.0.0'.
"""

import os
from pathlib import Path
from typing import Optional

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio is required for the UI. Install with: pip install 'salesbench[ui]'")

from salesbench.storage.json_writer import JSONResultsWriter


def create_leaderboard(
    results_dir: Optional[str] = None,
    title: str = "SalesBench Leaderboard",
    refresh_interval: int = 60,
) -> gr.Blocks:
    """Create Gradio leaderboard interface.

    Args:
        results_dir: Directory containing JSON result files.
                    Defaults to 'results' in current directory.
        title: Title displayed on the leaderboard.
        refresh_interval: Seconds between data refreshes.

    Returns:
        Gradio Blocks app.
    """
    if results_dir is None:
        results_dir = os.getenv("SALESBENCH_RESULTS_DIR", "results")

    writer = JSONResultsWriter(results_dir)

    def load_leaderboard_data():
        """Load and format data for the leaderboard table."""
        data = writer.get_leaderboard_data()

        # Sort by mean score (descending)
        data = sorted(data, key=lambda x: x.get("mean_score", 0), reverse=True)

        # Format for display
        rows = []
        for i, r in enumerate(data, 1):
            rows.append(
                [
                    i,  # Rank
                    r.get("model", "unknown"),
                    r.get("domain", "insurance"),
                    f"{r.get('mean_score', 0):.1f}",
                    f"{r.get('std_score', 0):.1f}",
                    f"{r.get('acceptance_rate', 0):.1%}",
                    r.get("episodes", 0),
                    r.get("timestamp", "")[:10] if r.get("timestamp") else "",
                ]
            )

        return rows

    def load_result_details(benchmark_id: str) -> str:
        """Load detailed results for a specific benchmark."""
        if not benchmark_id:
            return "Select a benchmark to view details."

        result = writer.get_result_by_id(benchmark_id)
        if not result:
            return f"Benchmark {benchmark_id} not found."

        config = result.get("config", {})
        aggregate = result.get("aggregate_metrics", {})

        details = f"""## Benchmark: {result.get('benchmark_id', 'unknown')}

### Configuration
- **Seller Model**: {config.get('seller_model', 'unknown')}
- **Buyer Model**: {config.get('buyer_model', 'unknown')}
- **Domain**: {config.get('domain', 'insurance')}
- **Episodes**: {result.get('completed_episodes', 0)}
- **Mode**: {config.get('mode', 'unknown')}

### Results
- **Mean Score**: {aggregate.get('mean_score', 0):.2f} (std: {aggregate.get('std_score', 0):.2f})
- **Acceptance Rate**: {aggregate.get('mean_acceptance_rate', 0):.1%}
- **Mean Calls**: {aggregate.get('mean_calls', 0):.1f}
- **Mean Offers**: {aggregate.get('mean_offers', 0):.1f}
- **Duration**: {result.get('duration_seconds', 0):.1f}s

### Timing
- **Started**: {result.get('started_at', 'unknown')}
- **Completed**: {result.get('completed_at', 'unknown')}
"""
        return details

    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(),
        css="""
        .leaderboard-table { font-size: 14px; }
        .rank-1 { background-color: #ffd700 !important; }
        .rank-2 { background-color: #c0c0c0 !important; }
        .rank-3 { background-color: #cd7f32 !important; }
        """,
    ) as demo:
        gr.Markdown(f"""
# {title}

Benchmarking AI agents on sales conversations.

[GitHub](https://github.com/Hamza-Mos/salesbench) | [Documentation](#docs)
        """)

        with gr.Tabs():
            with gr.Tab("Leaderboard"):
                leaderboard_table = gr.Dataframe(
                    value=load_leaderboard_data,
                    headers=[
                        "Rank",
                        "Model",
                        "Domain",
                        "Score",
                        "Std",
                        "Accept Rate",
                        "Episodes",
                        "Date",
                    ],
                    datatype=["number", "str", "str", "str", "str", "str", "number", "str"],
                    every=refresh_interval,
                    interactive=False,
                    elem_classes=["leaderboard-table"],
                )

                gr.Markdown("""
### Metrics Explained
- **Score**: Composite score based on sales outcomes (higher is better)
- **Accept Rate**: Percentage of offers accepted by buyers
- **Episodes**: Number of independent sales sessions evaluated
                """)

            with gr.Tab("How to Submit"):
                gr.Markdown("""
## Submitting Results

### 1. Run the Benchmark

```bash
# Install salesbench
pip install salesbench

# Run benchmark with your model
salesbench run-benchmark \\
    --seller-model YOUR_MODEL \\
    --domain insurance \\
    --mode production \\
    -n 100
```

### 2. Find Your Results

Results are saved to the `results/` directory as JSON files:
```
results/
  bench_abc123_20240115_143022.json
```

### 3. Submit

**Option A: GitHub PR**
1. Fork the repository
2. Add your results file to `results/`
3. Submit a pull request

**Option B: Upload**
Contact us to add your results directly.

### Requirements
- Minimum 100 episodes for official rankings
- Use `--mode production` for comparable results
- Include model name and configuration
                """)

            with gr.Tab("About"):
                gr.Markdown("""
## About SalesBench

SalesBench is a benchmark for evaluating AI agents on sales conversations.
It provides a standardized environment where agents must:

1. **Search and prioritize leads** using CRM tools
2. **Engage in sales calls** with simulated buyers
3. **Understand customer needs** and propose appropriate products
4. **Handle objections** and close deals

### Domains

Currently supported domains:
- **Insurance**: Life insurance cold-calling

### Evaluation

Agents are scored on:
- Successful sales (accepted offers)
- Sales efficiency (time and calls used)
- Customer satisfaction (objection handling)
- Compliance (DNC list violations)

### Technical Details

- Built on the [Verifiers](https://github.com/primeintellect-ai/verifiers) framework
- Deterministic persona generation for reproducibility
- Multi-provider LLM support (OpenAI, Anthropic, etc.)
                """)

        gr.Markdown(
            """
---
*Last updated every {} seconds*
        """.format(refresh_interval)
        )

    return demo


def main():
    """Launch the leaderboard app."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch SalesBench Leaderboard")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing result files (default: results/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )

    args = parser.parse_args()

    demo = create_leaderboard(results_dir=args.results_dir)
    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
