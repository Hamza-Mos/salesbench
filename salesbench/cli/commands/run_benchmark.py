"""Run benchmark command for CLI.

Runs N episodes in parallel with Supabase/telemetry integrations.
"""

import argparse
import os

from salesbench.cli.commands import register_command


@register_command("run-benchmark")
def run_benchmark_command(args: argparse.Namespace) -> int:
    """Run a benchmark with multiple episodes."""
    from salesbench.models import (
        DEFAULT_BUYER_MODEL,
        get_default_benchmark_models,
        parse_model_list,
        parse_model_spec,
    )
    from salesbench.runner import BenchmarkConfig, BenchmarkRunner

    # Determine which models to benchmark
    if args.models:
        models = parse_model_list(args.models)
    else:
        # No models specified - use default benchmark set
        models = get_default_benchmark_models()
        print("No --models specified. Using default set:")
        for m in models:
            print(f"  - {m}")

    # Parse buyer model
    buyer_model = (
        args.buyer_model or os.environ.get("SALESBENCH_BUYER_MODEL") or DEFAULT_BUYER_MODEL
    )
    buyer_spec = parse_model_spec(buyer_model)

    # Run benchmarks for all models
    print(f"\nRunning benchmark for {len(models)} model(s)")
    print(f"Buyer model: {buyer_spec}")
    print("=" * 60)

    all_passed = True
    for i, model_spec in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Benchmarking: {model_spec}")
        print("-" * 40)

        config = BenchmarkConfig.from_cli_args(
            mode=args.mode,
            episodes=args.episodes,
            seed=args.seed,
            leads=args.leads,
            safety_max_turns=args.safety_max_turns,
            parallelism=args.parallelism,
            seller_model=str(model_spec),
            buyer_model=str(buyer_spec),
            no_supabase=args.no_supabase,
            enable_telemetry=args.telemetry,
            output=args.output,
            verbose=args.verbose,
            name=args.name or "",
            domain=args.domain,
            hours=args.hours,
            progress_interval=args.progress_interval,
        )

        runner = BenchmarkRunner(config)
        result = runner.run()

        if result.failed_episodes > 0:
            all_passed = False

    print("\n" + "=" * 60)
    print(f"Completed benchmarks for {len(models)} model(s)")

    return 0 if all_passed else 1
