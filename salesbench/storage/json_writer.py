"""JSON file storage for SalesBench benchmark results.

Simple, portable storage using JSON files. Results are stored in a
local directory and can be easily shared, version controlled, or
uploaded to leaderboards.

Directory structure:
    results/
        2024-01-20_14-30-45_gpt-4o/
            summary.json      # Always: config, metrics, episode summaries (no traces)
            traces.json       # Only with -v: full conversation trajectories
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JSONResultsWriter:
    """JSON file storage for benchmark results.

    Each benchmark run creates a timestamped directory containing:
    - summary.json: Configuration, aggregate metrics, episode results (without traces)
    - traces.json: Full conversation trajectories (only if verbose=True)

    Directory naming: {timestamp}_{model_name}/
    """

    def __init__(self, results_dir: Path | str = "results"):
        """Initialize the JSON results writer.

        Args:
            results_dir: Base directory to store result files.
                        Created if it doesn't exist.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write_benchmark(
        self,
        benchmark_id: str,
        result: dict,
        include_traces: bool = False,
        custom_name: Optional[str] = None,
    ) -> Path:
        """Write benchmark result to JSON files in a timestamped directory.

        Args:
            benchmark_id: Unique identifier for the benchmark run.
            result: Dictionary containing benchmark results.
            include_traces: If True, also write traces.json with full trajectories.
            custom_name: Optional custom directory name (from -o flag).

        Returns:
            Path to the created directory.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Extract model name for directory naming
        config = result.get("config", {})
        seller_model = config.get("seller_model", "unknown")
        # Clean model name for filesystem (e.g., "openai/gpt-4o" -> "gpt-4o")
        model_name = seller_model.split("/")[-1] if "/" in seller_model else seller_model
        model_name = re.sub(r"[^\w\-.]", "_", model_name)

        # Create directory name
        if custom_name:
            dir_name = f"{timestamp}_{custom_name}"
        else:
            dir_name = f"{timestamp}_{model_name}"

        run_dir = self.results_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Prepare summary (without traces)
        summary_result = self._make_serializable(result)
        if "episode_results" in summary_result:
            for ep in summary_result["episode_results"]:
                # Remove trajectory from summary, keep everything else
                trajectory = ep.pop("trajectory", [])

        # Write summary.json (always)
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_result, f, indent=2, default=str)
        logger.info(f"Wrote benchmark summary to {summary_path}")

        # Write traces.json (only if verbose)
        if include_traces:
            traces_result = self._make_serializable(result)
            # Keep only trajectories and episode identifiers
            traces_data = {
                "benchmark_id": traces_result.get("benchmark_id"),
                "episodes": [],
            }
            for ep in traces_result.get("episode_results", []):
                traces_data["episodes"].append(
                    {
                        "episode_id": ep.get("episode_id"),
                        "episode_index": ep.get("episode_index"),
                        "seed": ep.get("seed"),
                        "trajectory": ep.get("trajectory", []),
                    }
                )

            traces_path = run_dir / "traces.json"
            with open(traces_path, "w") as f:
                json.dump(traces_data, f, indent=2, default=str)
            logger.info(f"Wrote traces to {traces_path}")

        return run_dir

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "to_dict"):
            return self._make_serializable(obj.to_dict())
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

    def list_results(self) -> list[Path]:
        """List all result files, sorted by modification time (newest first).

        Returns:
            List of paths to summary.json files in result directories.
        """
        return sorted(
            self.results_dir.glob("*/summary.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def load_result(self, path: Path) -> dict:
        """Load result from JSON file.

        Args:
            path: Path to the JSON result file.

        Returns:
            Parsed result dictionary.
        """
        with open(path) as f:
            return json.load(f)

    def load_latest(self) -> Optional[dict]:
        """Load the most recent result file.

        Returns:
            Parsed result dictionary, or None if no results exist.
        """
        results = self.list_results()
        if not results:
            return None
        return self.load_result(results[0])

    def get_leaderboard_data(self) -> list[dict]:
        """Load all results formatted for leaderboard display.

        Returns:
            List of dictionaries with leaderboard-friendly fields.
        """
        results = []

        for path in self.list_results():
            try:
                data = self.load_result(path)
                config = data.get("config", {})
                aggregate = data.get("aggregate_metrics", {})
                token_usage = aggregate.get("total_token_usage", {})
                cost_breakdown = aggregate.get("total_cost_breakdown", {})

                # Calculate total tokens
                total_tokens = (
                    token_usage.get("seller_input_tokens", 0)
                    + token_usage.get("seller_output_tokens", 0)
                    + token_usage.get("buyer_input_tokens", 0)
                    + token_usage.get("buyer_output_tokens", 0)
                )

                results.append(
                    {
                        "file": path.name,
                        "benchmark_id": data.get("benchmark_id", "unknown"),
                        "model": config.get("seller_model", "unknown"),
                        "buyer_model": config.get("buyer_model", "unknown"),
                        "domain": config.get("domain", "insurance"),
                        "episodes": data.get("completed_episodes", 0),
                        "mean_score": aggregate.get("mean_score", 0),
                        "std_score": aggregate.get("std_score", 0),
                        "acceptance_rate": aggregate.get("mean_acceptance_rate", 0),
                        "mean_calls": aggregate.get("mean_calls", 0),
                        "mean_offers": aggregate.get("mean_offers", 0),
                        "timestamp": data.get("started_at", ""),
                        "duration_seconds": data.get("duration_seconds", 0),
                        # New metrics
                        "total_tokens": total_tokens,
                        "total_cost": cost_breakdown.get("total_cost", 0),
                        "mean_action_based_minutes": aggregate.get("mean_action_based_minutes", 0),
                        "mean_token_based_minutes": aggregate.get("mean_token_based_minutes", 0),
                        "mean_conversation_turns": aggregate.get("mean_conversation_turns", 0),
                        "total_dnc_violations": aggregate.get("total_dnc_violations", 0),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

        return results

    def get_result_by_id(self, benchmark_id: str) -> Optional[dict]:
        """Find and load result by benchmark ID.

        Args:
            benchmark_id: The benchmark ID to search for.

        Returns:
            Parsed result dictionary, or None if not found.
        """
        for path in self.list_results():
            try:
                data = self.load_result(path)
                if data.get("benchmark_id", "").startswith(benchmark_id):
                    return data
            except Exception:
                continue
        return None

    def _find_result_dir(self, benchmark_id: str) -> Optional[Path]:
        """Find the result directory for a benchmark ID.

        Args:
            benchmark_id: The benchmark ID to search for.

        Returns:
            Path to the result directory, or None if not found.
        """
        for path in self.list_results():
            try:
                data = self.load_result(path)
                if data.get("benchmark_id", "").startswith(benchmark_id):
                    return path.parent
            except Exception:
                continue
        return None

    def has_traces(self, benchmark_id: str) -> bool:
        """Check if traces.json exists for a benchmark.

        Args:
            benchmark_id: The benchmark ID to check.

        Returns:
            True if traces.json exists, False otherwise.
        """
        result_dir = self._find_result_dir(benchmark_id)
        if result_dir is None:
            return False
        traces_path = result_dir / "traces.json"
        return traces_path.exists()

    def load_traces(self, benchmark_id: str) -> Optional[dict]:
        """Load traces.json for a benchmark.

        Args:
            benchmark_id: The benchmark ID to load traces for.

        Returns:
            Parsed traces dictionary, or None if not found.
        """
        result_dir = self._find_result_dir(benchmark_id)
        if result_dir is None:
            return None
        traces_path = result_dir / "traces.json"
        if not traces_path.exists():
            return None
        try:
            with open(traces_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load traces from {traces_path}: {e}")
            return None

    def delete_result(self, path: Path) -> bool:
        """Delete a result file.

        Args:
            path: Path to the result file.

        Returns:
            True if deleted, False if not found.
        """
        try:
            path.unlink()
            logger.info(f"Deleted result file: {path}")
            return True
        except FileNotFoundError:
            return False

    def export_leaderboard_csv(self, output_path: Optional[Path] = None) -> Path:
        """Export leaderboard data to CSV.

        Args:
            output_path: Path for the CSV file. Defaults to
                        results_dir/leaderboard.csv.

        Returns:
            Path to the created CSV file.
        """
        import csv

        if output_path is None:
            output_path = self.results_dir / "leaderboard.csv"

        data = self.get_leaderboard_data()

        if not data:
            logger.warning("No results to export")
            return output_path

        fieldnames = list(data[0].keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"Exported leaderboard to {output_path}")
        return output_path
