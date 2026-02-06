# SalesBench Prime RL Harness

Standalone, Prime-stack-first harness for training and evaluating sales agents with real model inference.

## What This Repository Is

This directory is a self-contained mini-repo built on Prime Intellect `verifiers`:

- `StatefulToolEnv` with a canonical runtime (`SalesEpisodeRuntime`)
- tool-only interaction (CRM, calling, callbacks, product quoting)
- reward objective is raw converted MRR (`revenue_mrr`); compliance/quality signals are emitted as metrics
- direct compatibility with:
  - `prime eval run`
  - `prime rl run`

## Directory Layout

- `salesbench_prime_rl/`: environment package
- `configs/lab/salesbench-prime-rl.toml`: hosted RL template
- `.env.example`: optional key template
- `Makefile`: easiest run interface (`make smoke`, `make benchmark`, etc.)

## Setup

```bash
cd /Users/hamza/Desktop/salesbench/salesbench_prime_rl_harness
uv sync
uv pip install -e .
uv tool install -U prime
prime login
prime env install .
```

Optional local env file:

```bash
cp .env.example .env
```

When `.env` is needed:

- Not required for default Prime Inference usage (`prime eval run ... -m openai/...`).
- Required only if you pass custom endpoint flags (for example `EXTRA_FLAGS="-k OPENAI_API_KEY -b https://api.openai.com/v1"`).

## Easiest Way To Run

Use Make targets (they call `prime eval run` directly):

```bash
make setup
make show-config
make smoke
make benchmark
make benchmark-tiny
make tui
```

`make tui` opens the evaluation browser for runs you already executed. Typical flow:

1. run `make eval` or `make benchmark`
2. then run `make tui`

## Original Harness Defaults (Copied Exactly)

Source of truth used:

- `/Users/hamza/Desktop/salesbench/salesbench/runner/config.py` (`MODE_PRESETS`)
- `/Users/hamza/Desktop/salesbench/salesbench/models.py` (`DEFAULT_BENCHMARK_MODELS`)

Mode presets from original harness:

| Mode | Episodes | Leads | Total Hours | Parallelism | safety_max_turns |
|---|---:|---:|---:|---:|---|
| `production` | 100 | 100 | 80 | 1 | `None` |
| `demo` | 3 | 50 | 2 | 1 | `None` |
| `test` | 3 | 5 | 16 | 1 | `None` |
| `debug` | 1 | 5 | 4 | 1 | `None` |

Notes:

- Original CLI help text says `demo (10 eps)`, but runtime `MODE_PRESETS` uses `3`; this harness follows `MODE_PRESETS`.
- Original benchmark model defaults were:
  - `openai/gpt-5.2`
  - `anthropic/claude-opus-4-5-20251101`
  - `google/gemini-3-pro-preview`
- Environment-level defaults are also aligned to original production baseline:
  - `seed=42`, `num_leads=100`, `work_days=10`, `hours_per_day=8`
  - legacy `safety_max_turns=None` is represented by omitting `MAX_TURNS` in Make targets.

## Mode-Based Commands

Run with original production defaults:

```bash
make benchmark MODE=production
```

Run a quick debug smoke:

```bash
make smoke MODE=debug
```

Demo mode is supported:

```bash
make benchmark MODE=demo
```

Run one model with test defaults:

```bash
make eval MODE=test MODEL=openai/gpt-4.1-mini
```

Split meanings:

- `train`: larger training-oriented scenarios/curriculum
- `eval`: stable comparison split for benchmark reporting (default)
- `test`: holdout-style validation split

## Useful Overrides

```bash
# Increase parallel episode execution
make benchmark MODE=production CONCURRENCY=4

# Override episode count
make benchmark MODE=production EPISODES=25

# Add explicit safety cap (legacy default was None)
make eval MODE=production MAX_TURNS=120

# Route to custom OpenAI-compatible endpoint
make eval MODEL=gpt-4o-mini EXTRA_FLAGS="-k OPENAI_API_KEY -b https://api.openai.com/v1"

# Tiny end-to-end sweep (convenience target)
make benchmark-tiny
```

## Prime CLI Direct (No Makefile)

```bash
prime eval run salesbench-prime-rl \
  -m openai/gpt-4.1-mini \
  -n 25 \
  -r 1 \
  -c 4 \
  -a '{"split":"eval","seed":42,"num_leads":100,"work_days":10,"hours_per_day":8,"num_examples":25,"eval_num_examples":25}' \
  --skip-upload
```

## Hosted RL

1. Edit `/Users/hamza/Desktop/salesbench/salesbench_prime_rl_harness/configs/lab/salesbench-prime-rl.toml` and set your environment IDs.
2. Run:

```bash
make rl
```

## Push To Prime Hub (Optional)

Push this environment to your private/team space:

```bash
prime env push --path . -v PRIVATE
```

After push, you can evaluate by Hub ID instead of local package name.

## Expected Warnings (Normal)

- `No upstream environment found.` means you are running locally (not from a pushed Hub environment).
- `No local endpoint registry found at ./configs/endpoints.py` is normal if you are using default Prime Inference and did not add custom endpoint aliases.
