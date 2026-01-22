# SalesBench

A benchmark for evaluating LLM agents on sales conversations. Built for [Prime Intellect Verifiers](https://github.com/primeintellect-ai/verifiers).

## Quick Start

```bash
pip install -e ".[all]"
cp .env.example .env  # Add your API key(s)

# Run benchmark
salesbench run-benchmark --models openai/gpt-4o --mode test -v
```

## Features

- **Multi-provider support**: OpenAI, Anthropic, Google, OpenRouter, Together, xAI
- **Reproducible**: Seeded persona generation for consistent evaluation
- **12 seller tools**: CRM, Calendar, Calling, Products
- **LLM buyer simulation**: Realistic buyer behavior
- **Batch benchmarking**: Run multiple models in one command
- **Natural termination**: No artificial turn limits - episodes end naturally

## How It Works

SalesBench simulates a realistic sales environment where an AI agent works as an insurance salesperson over a configurable business period.

### Time System

Each episode simulates business days of work (default: 10 days, 8 hours/day):

```
+-------------------------------------------------------------+
|  EPISODE TIME BUDGET (configurable)                          |
|                                                              |
|  Default: 10 days x 8 hours/day = 80 working hours          |
|  Working hours: 9 AM to (9 + hours_per_day)                 |
|                                                              |
|  Time advances with actions (configurable in BudgetConfig): |
|  - start_call    -> +1 minute (start_call_cost)             |
|  - propose_plan  -> +4 minutes (propose_plan_cost)          |
|  - search_leads  -> +1.0 minutes (search_cost)              |
|  - conversation  -> +2 minutes (conversation_turn_cost)     |
|  - Other tools   -> no time cost                            |
|                                                              |
|  Override via CLI: --days 5 --hours-per-day 4               |
+-------------------------------------------------------------+
```

### Time Model Configuration

SalesBench supports two time models for tracking budget usage:

| Model | Description | Use Case |
|-------|-------------|----------|
| `action` (default) | Fixed cost per action | Consistent, predictable timing |
| `token` | Time based on token consumption (~150 tokens/minute) | Reflects actual LLM usage |

**Both metrics are always tracked** regardless of which model is active:

```python
from salesbench.core.config import BudgetConfig

# Default: action-based time
config = BudgetConfig()

# Token-based time
config = BudgetConfig(time_model="token")

# Custom action time costs
config = BudgetConfig(
    start_call_cost=1.0,      # Minutes to initiate a call (default: 1.0)
    propose_plan_cost=4.0,    # Minutes to present offer + get response (default: 4.0)
    search_cost=1.0,          # Minutes for CRM search (default: 1.0)
    conversation_turn_cost=2.0,  # Minutes per conversation turn (default: 2.0)
)

# Custom token rate (for token-based time model)
config = BudgetConfig(time_model="token", tokens_per_minute=200.0)
```

### Output Metrics

Results include both time metrics for analysis:

```json
{
  "action_based_minutes": 245,
  "token_based_minutes": 312.5,
  "time_model_used": "action",
  "budget_minutes_used": 245,
  "conversation_turns": 142,
  "total_calls": 47
}
```

| Metric | Description |
|--------|-------------|
| `action_based_minutes` | Time from fixed action costs |
| `token_based_minutes` | Time estimated from token usage |
| `time_model_used` | Which model was used for budget (`action` or `token`) |
| `budget_minutes_used` | The active model's minutes (used for budget limits) |
| `conversation_turns` | Total seller-buyer conversation exchanges during calls |

### Episode Flow

```
+-------------------------------------------------------------+
|                     EPISODE START                            |
|              Day 1, 9:00 AM, 100 leads available            |
+-------------------------------------------------------------+
                              |
                              v
            +---------------------------------------+
            |           AGENT TURN                  |
            |                                       |
            |  Tools available:                     |
            |  - crm.search_leads                   |
            |  - crm.get_lead                       |
            |  - calling.start_call  (+1 min)       |
            |  - calling.propose_plan (+4 min)      |
            |  - calling.end_call                   |
            |  - products.list_plans                |
            |  - products.quote_premium             |
            +---------------------------------------+
                              |
                              v
            +---------------------------------------+
            |       BUYER SIMULATOR                 |
            |                                       |
            |  LLM-powered realistic responses:     |
            |  - Accept -> CONVERTED                |
            |  - Reject -> continue negotiating     |
            |  - "Stop calling" -> DNC              |
            |  - Hang up -> end call                |
            +---------------------------------------+
                              |
                              v
            +---------------------------------------+
            |     CHECK NATURAL TERMINATION         |
            |                                       |
            |  1. NO_LEADS - all resolved           |
            |  2. TIME_LIMIT - day 10 ended         |
            |                                       |
            |  No artificial turn limits!           |
            +---------------------------------------+
                              |
                  +-----------+-----------+
                  v                       v
            [Continue]              [Episode End]
                  |                       |
                  +-------+               v
                          |     +-----------------+
                          |     | FINAL SCORE     |
                          |     | = Total Revenue |
                          |     | (sum of monthly |
                          +-----| premiums)       |
                                +-----------------+
```

### Lead Lifecycle

```
  ACTIVE ----+----> CONVERTED (accepted a plan)
             |
             +----> DNC (buyer requested "do not call")
```

### Reproducibility

SalesBench is designed for **reproducible evaluation**, following principles from [τ-bench](https://github.com/sierra-research/tau-bench) and [τ²-bench](https://github.com/sierra-research/tau2-bench):

#### Why Reproducibility Matters

LLM benchmarks face inherent challenges with non-determinism:
- LLM outputs vary even with the same prompt
- User simulators add another layer of randomness
- Results can be inconsistent across runs

SalesBench addresses these with a multi-layered approach:

| Component | Strategy | Effect |
|-----------|----------|--------|
| **Lead Generation** | Seeded RNG | Identical leads for same seed |
| **Buyer LLM** | Temperature = 0.0 | Deterministic buyer responses |
| **Seller LLM** | Temperature = 0.0 (default) | Consistent agent behavior |
| **Buyer Model** | Fixed (configurable) | Same buyer persona across runs |

#### Temperature Settings

```bash
# Default: fully deterministic (τ-bench style)
SALESBENCH_BUYER_TEMPERATURE=0.0   # Buyer is deterministic
# Seller uses temperature=0.0 by default

# For more natural conversations (optional)
SALESBENCH_BUYER_TEMPERATURE=0.3   # Slight variation
```

Like [τ²-bench](https://arxiv.org/abs/2406.12045), we use **temperature 0** to promote deterministic outputs. This means:
- Same seed + same model = nearly identical results
- Fair A/B comparison between models
- Debuggable conversation traces

#### Static Buyer Simulator

The buyer LLM uses a **fixed system prompt** and **consistent persona** derived from the seeded lead data:

```
Buyer Simulator
├── System prompt (static, never changes)
├── Persona attributes (from seeded generation)
│   ├── Name, age, income, job
│   ├── Life trigger (why they need insurance)
│   └── Objection style (direct, price-focused, etc.)
└── Hidden state (trust, interest, patience, threshold)
```

The buyer simulator doesn't learn or adapt between episodes—it's a consistent evaluation oracle.

### Persona Generation & Seeding

SalesBench generates **reproducible leads** using a seeded random number generator. This ensures:
- Same seed = identical leads across runs
- Fair comparison between different models
- Reproducible benchmark results

#### How Seeding Works

```
Episode Seed = base_seed + episode_index

Example with base_seed=42:
  Episode 0: seed=42  → generates leads L1, L2, L3...
  Episode 1: seed=43  → generates leads L1', L2', L3'...
  Episode 2: seed=44  → generates leads L1'', L2'', L3''...
```

Each episode's seed determines:
1. Which archetypes are selected
2. Specific attribute values (age, income, etc.)
3. Hidden buyer state (trust, interest, patience)
4. Lead temperature distribution

#### Archetypes

Leads are generated from **10 diverse archetypes**:

| Archetype | Age Range | Income Range | Key Traits |
|-----------|-----------|--------------|------------|
| Young Professional | 25-35 | $50K-$120K | New job, first home, getting married |
| New Parent | 28-42 | $60K-$150K | Family-focused, price-conscious |
| Mid-Career Professional | 35-50 | $80K-$200K | Career advancement, estate planning |
| Pre-Retiree | 50-65 | $100K-$300K | Retirement planning, legacy concerns |
| Small Business Owner | 30-55 | $75K-$250K | Business protection, key person insurance |
| Healthcare Worker | 25-55 | $45K-$180K | Disability concerns, income protection |
| Blue Collar Worker | 25-55 | $35K-$80K | Price-focused, trust issues |
| High Net Worth | 40-65 | $250K-$500K | Estate tax, wealth transfer |
| Single Parent | 28-50 | $40K-$100K | Sole provider anxiety, high interest |
| Skeptic | 30-60 | $50K-$150K | Data-driven, low trust |

#### Temperature Distribution

Lead "temperature" indicates buyer interest level:

| Temperature | Probability | Behavior |
|-------------|-------------|----------|
| HOT | 3% | Ready to buy, asks good questions |
| WARM | 12% | Interested but cautious |
| LUKEWARM | 35% | Skeptical, needs convincing |
| COLD | 40% | Not interested, short responses |
| HOSTILE | 10% | Annoyed, will end call quickly |

#### Hidden State

Each lead has hidden attributes the seller cannot see directly:

| Attribute | Range | Description |
|-----------|-------|-------------|
| `trust` | 0.0-1.0 | Trust in salespeople (affects acceptance) |
| `interest` | 0.0-1.0 | Interest in insurance (affects engagement) |
| `patience` | 0.0-1.0 | How long before hanging up (decays ~12% per rejection, warns seller when low) |
| `close_threshold` | 0.01-0.15 | Max premium as % of monthly income |

The buyer LLM uses these hidden values to make realistic decisions.

#### Buyer Patience & Frustration

Buyer patience is a critical hidden state that affects call outcomes:

| Event | Effect |
|-------|--------|
| Rejection | -12% patience (base decay) |
| 3+ rejections | -18% patience (increased frustration) |
| Patience ≤ 20% | Warning included in tool result |
| Patience ≤ 5% | Automatic hang-up |
| Patience = 0% | Hang-up + DNC request |

This models realistic buyer behavior where repeated unwanted offers lead to frustration and call termination.

#### Preview Leads

```bash
# Preview leads for a specific seed
salesbench seed-leads --seed 42 --count 10

# Show full hidden state (debug)
salesbench seed-leads --seed 42 --count 5 --show-hidden
```

### Natural Termination (No Turn Limits)

Unlike benchmarks with artificial turn limits, SalesBench lets episodes end naturally:

| Reason | What Triggers It | Why It's Natural |
|--------|------------------|------------------|
| `NO_LEADS` | All leads are converted/DNC | No more prospects to work |
| `TIME_LIMIT` | Simulated time passes `total_days` | Business period ends |

This tests whether agents can complete long-horizon tasks without artificial constraints.

### Context Compaction

Long episodes can exceed model context windows. SalesBench uses **LLM-based compaction** to handle this naturally.

#### How It Works

```
1. Context exceeds token threshold (80% of available context)
2. Split: older_messages | recent_messages (keep last N verbatim)
3. Send older_messages to compaction prompt
4. LLM returns structured memory summary
5. Replace context: [memory_summary] + [recent_messages]
```

#### Why LLM-Based Compaction?

The benchmark tests the model's **natural ability to manage context**—not hardcoded heuristics:

| Old Approach (Removed) | New Approach (LLM-Based) |
|------------------------|--------------------------|
| Hardcoded observation masking | Model decides what's important |
| Priority-based retention | Model extracts key facts |
| FIFO turn removal | Model summarizes coherently |

#### Configuration

Each model has a `compaction_keep_recent` setting that determines how many recent messages to keep verbatim:

```python
# salesbench/models.py
SUPPORTED_MODELS = {
    "gpt-4o-mini": ModelConfig(
        128_000, 16_384, "openai",
        compaction_keep_recent=6,   # Smaller model, fewer recent
    ),
    "gpt-5.2": ModelConfig(
        400_000, 128_000, "openai",
        compaction_keep_recent=15,  # Large context, more recent
    ),
    "claude-opus-4-5-20251101": ModelConfig(
        200_000, 64_000, "anthropic",
        compaction_keep_recent=15,
    ),
}
```

#### What's Preserved

| Component | Behavior |
|-----------|----------|
| **AnchoredState** | Always injected fresh (leads found, decisions, active call) |
| **Recent N messages** | Kept verbatim for coherent responses |
| **Older messages** | Summarized by LLM into memory bullets |

#### Compaction Prompts

**Seller compaction** summarizes:
- Conversations with each lead (discussed, objections, offers)
- Lead responses and decisions
- Key facts learned from tools
- Patterns (what works, what doesn't)

**Buyer compaction** summarizes:
- Offers received and responses
- Budget/affordability statements made
- Objections and concerns raised
- Trust signals about the seller

## Running Benchmarks

```bash
# Single model
salesbench run-benchmark --models openai/gpt-4o --mode production

# Multiple models
salesbench run-benchmark --models openai/gpt-4o,anthropic/claude-sonnet-4-20250514 --mode production

# Run default benchmark set (6 models)
salesbench run-benchmark --mode production
```

### Benchmark Modes

| Mode | Episodes | Leads | Days | Hours/Day | Purpose |
|------|----------|-------|------|-----------|---------|
| `production` | 100 | 100 | 10 | 8 | Official benchmark |
| `demo` | 5 | 20 | 2 | 8 | Quick showcase |
| `test` | 3 | 5 | 2 | 8 | Fast validation |
| `debug` | 1 | 5 | 1 | 4 | Quick iteration |

All modes use natural termination by default (no safety limits). Episodes end via time limit, lead exhaustion, or agent-initiated quit.

### CLI Options

```bash
# Override mode presets
salesbench run-benchmark --mode demo --episodes 10 --leads 30
salesbench run-benchmark --mode test --days 1 --hours-per-day 4

# Full option list
salesbench run-benchmark \
  --models openai/gpt-4o \
  --mode production \
  --episodes 100 \        # Override number of episodes
  --leads 100 \           # Override leads per episode
  --days 10 \             # Override simulated business days
  --hours-per-day 8 \     # Override working hours per day
  --seed 42 \             # Base random seed
  --parallelism 4 \       # Concurrent episodes
  --safety-max-turns 300  # Optional turn limit (default: none)
```

### Configuration Files

To customize beyond CLI options, edit these files:

| What to Change | File | Class/Variable |
|----------------|------|----------------|
| **Mode presets** (episodes, leads, days, hours) | `salesbench/runner/config.py` | `MODE_PRESETS` |
| **Time costs** (action costs, tokens/min) | `salesbench/core/config.py` | `BudgetConfig` |
| **Persona distribution** (temperature %, age/income ranges) | `salesbench/core/config.py` | `PersonaGenerationConfig` |
| **Context compaction** (recent messages to keep) | `salesbench/models.py` | `ModelConfig.compaction_keep_recent` |

#### Example: Custom Time Costs

```python
# salesbench/core/config.py - BudgetConfig defaults
@dataclass
class BudgetConfig:
    total_days: int = 10              # Simulated business days
    hours_per_day: int = 8            # Working hours (9 AM to 5 PM)
    time_model: str = "action"        # "action" or "token"
    conversation_turn_cost: float = 2.0   # Minutes per conversation turn
    tokens_per_minute: float = 150.0      # For token-based time

    # Action time costs (minutes)
    start_call_cost: float = 1.0      # Time to initiate a call
    search_cost: float = 1.0          # Time for CRM search
    propose_plan_cost: float = 4.0    # Time to present offer + get response
```

### Scoring

SalesBench uses **revenue-based scoring** following the pattern of industry benchmarks like τ-bench and WebArena:

**Score = Total Revenue** (sum of monthly premiums from accepted plans)

This is:
- **Interpretable**: "Agent earned $2,400 in monthly premiums"
- **Aligned with business goal**: Maximize revenue
- **No config needed**: Revenue is revenue

### Additional Metrics Tracked

Beyond the primary score, SalesBench tracks comprehensive metrics for analysis:

| Metric | Description |
|--------|-------------|
| DNC Violations | Calls to Do-Not-Call list (compliance tracking) |
| Acceptance Rate | Offers accepted / total offers |
| Conversion Rate | Accepts / total calls |
| Mean Calls | Average calls per episode |
| End Calls | Buyer hang-ups |
| Patience Warnings | Low patience alerts shown to seller |

All metrics are displayed in the UI leaderboard and saved to results JSON.

#### Example: Custom Mode Preset

```python
# salesbench/runner/config.py - Add or modify presets
MODE_PRESETS = {
    RunMode.DEBUG: {
        "num_episodes": 1,
        "num_leads": 5,
        "safety_max_turns": None,
        "parallelism": 1,
        "budget": {
            "total_days": 1,
            "hours_per_day": 4,  # Short 4-hour day for fast iteration
        },
    },
    # ... other modes
}
```

## Results

Results are saved to `results/` as JSON files. View the leaderboard:

```bash
salesbench leaderboard
```

## Documentation

- [Setup Guide](docs/SETUP.md) - API keys, providers, installation
- [Running Benchmarks](docs/RUNNING.md) - Examples, modes, batch runs
- [Adding Domains](docs/CONTRIBUTING.md) - Extend to new sales scenarios

## CLI Reference

```bash
salesbench run-benchmark   # Run benchmark (see options above)
salesbench run-episode     # Single episode (debug)
salesbench list-models     # Show known models
salesbench list-domains    # Show available domains
salesbench leaderboard     # Launch web UI
salesbench seed-leads      # Preview personas
salesbench inspect-products # View products
salesbench quote           # Get premium quote
```

For full options: `salesbench run-benchmark --help`

## Verifiers Integration

```python
import verifiers as vf

env = vf.load_environment("salesbench", seed=42, num_leads=100)
results = env.evaluate(client, model="gpt-4o")
```

## License

MIT
