# SalesBench

AI Social Intelligence Benchmark for evaluating agents through life insurance cold-calling scenarios. Built for Prime Intellect Verifiers.

## Features

- **LLM-based buyer simulation** - Realistic buyer behavior using any LLM provider
- **Multi-provider support** - OpenAI, Anthropic, OpenRouter, xAI (Grok), Together AI, Google Gemini
- **Seeded persona generation** - 100 reproducible leads per episode
- **12 seller tools** - CRM, Calendar, Calling, Products
- **RL-ready scoring** - Bounded rewards, serializable state
- **Prime Intellect Verifiers compatible** - StatefulToolEnv interface

---

## Quick Start (5 Minutes)

### Step 1: Install

```bash
git clone https://github.com/salesbench/salesbench.git
cd salesbench

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[all]"
```

### Step 2: Get an API Key (Choose ONE)

| Provider | Get Key | Free Tier? |
|----------|---------|------------|
| **OpenAI** | https://platform.openai.com/api-keys | $5 credit |
| **Anthropic** | https://console.anthropic.com/settings/keys | $5 credit |
| **OpenRouter** | https://openrouter.ai/keys | Some free models |
| **xAI (Grok)** | https://console.x.ai/ | Check availability |
| **Together AI** | https://api.together.xyz/settings/api-keys | $25 credit |
| **Google AI** | https://aistudio.google.com/app/apikey | Free tier |

### Step 3: Configure

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```bash
# Add ONE of these (whichever you have):
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here
# OR
OPENROUTER_API_KEY=sk-or-your-key-here
# OR
XAI_API_KEY=xai-your-key-here
# OR
TOGETHER_API_KEY=your-key-here
# OR
GOOGLE_API_KEY=your-key-here
```

### Step 4: Test It Works

```bash
# These don't need an API key:
salesbench seed-leads --seed 42 --n 5
salesbench inspect-products

# This needs an API key:
salesbench run-episode --seed 42 --leads 5 --max-turns 10 -v
```

---

## Prime Intellect Verifiers Integration

SalesBench is fully compatible with Prime Intellect Verifiers:

```python
import verifiers as vf

# Load via verifiers
env = vf.load_environment("salesbench", seed=42, num_leads=100)

# Or direct import
from salesbench import load_environment, SalesBenchToolEnv

env = load_environment(seed=42, num_leads=100)
assert isinstance(env, SalesBenchToolEnv)

# Run evaluation
results = env.evaluate(client, model="gpt-4o")
```

### Environment Hub

```bash
# Publish to Prime Intellect Environment Hub
prime login
prime env push
```

---

## Supported LLM Providers

SalesBench auto-detects which provider to use based on which API key is set.

| Provider | Env Variable | Default Model | Notes |
|----------|--------------|---------------|-------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | Best tool calling support |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022` | Excellent reasoning |
| OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` | Access 100+ models |
| xAI | `XAI_API_KEY` | `grok-beta` | Grok models |
| Together AI | `TOGETHER_API_KEY` | `meta-llama/Llama-3.1-70B-Instruct-Turbo` | Open source models |
| Google | `GOOGLE_API_KEY` | `gemini-1.5-flash` | Gemini models |

### Using a Specific Provider

```python
from salesbench import load_environment
from salesbench.llm import create_client

# Auto-detect provider
env = load_environment(seed=42)

# Or specify provider explicitly
client = create_client(provider="anthropic", model="claude-3-5-sonnet-20241022")
```

---

## Environment Variables Reference

### LLM Providers (Required: At least ONE)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `XAI_API_KEY` | xAI (Grok) API key |
| `TOGETHER_API_KEY` | Together AI API key |
| `GOOGLE_API_KEY` | Google AI API key |

### Model Configuration (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `SALESBENCH_BUYER_MODEL` | `gpt-4o-mini` | Model for buyer decisions |
| `SALESBENCH_BUYER_TEMPERATURE` | `0.3` | Buyer LLM temperature |
| `SALESBENCH_SELLER_MODEL` | `gpt-4o-mini` | Model for seller agent |

---

## CLI Commands

| Command | API Key? | Purpose |
|---------|----------|---------|
| `seed-leads` | No | Generate sample personas |
| `inspect-products` | No | Show insurance products |
| `quote` | No | Calculate premium |
| `run-episode` | **Yes** | Run full episode with LLM agents |
| `run-benchmark` | **Yes** | Run multiple episodes in parallel |

```bash
# Preview personas
salesbench seed-leads --seed 42 --n 10

# See products
salesbench inspect-products

# Calculate a quote
salesbench quote --plan TERM --age 35 --coverage 500000 --term 20

# Run episode (requires API key)
salesbench run-episode --seed 42 --leads 20 --max-turns 50 -v

# Run benchmark
salesbench run-benchmark --mode test --episodes 3 -v
```

---

## Available Tools

### CRM Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `crm_search_leads` | `temperature`, `min_income`, `max_age`, `limit` | Search leads |
| `crm_get_lead` | `lead_id` | Get lead details |
| `crm_update_lead` | `lead_id`, `notes`, `temperature` | Update lead |
| `crm_log_call` | `lead_id`, `call_id`, `outcome`, `notes` | Log call |

### Calendar Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `calendar_get_availability` | `day` | Get available slots |
| `calendar_schedule_call` | `lead_id`, `day`, `hour` | Schedule call |

### Calling Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `calling_start_call` | `lead_id` | Start call |
| `calling_propose_plan` | `plan_id`, `monthly_premium`, `coverage_amount`, `next_step` | Present offer |
| `calling_end_call` | `reason` | End call |

### Product Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `products_list_plans` | - | List all plans |
| `products_get_plan` | `plan_id` | Get plan details |
| `products_quote_premium` | `plan_id`, `age`, `coverage_amount`, `risk_class`, `term_years` | Calculate premium |

---

## Scoring

| Event | Points | Notes |
|-------|--------|-------|
| Plan accepted | +100 | Base reward |
| Close now bonus | +50 | If `next_step=close_now` |
| Schedule followup bonus | +20 | If `next_step=schedule_followup` |
| Premium multiplier | +0.5 x premium | Higher premiums = more reward |
| Plan rejected | -5 | Per rejection |
| Buyer ends call | -10 | Buyer hangs up |
| DNC violation | -200 | Calling someone on Do Not Call list |

---

## Project Structure

```
salesbench/
├── salesbench/                   # Main package
│   ├── __init__.py               # Package exports (load_environment, SalesBenchToolEnv)
│   ├── __main__.py               # Entry point for `python -m salesbench`
│   ├── environment.py            # SalesBenchToolEnv (verifiers StatefulToolEnv)
│   │
│   ├── core/                     # Core types and configuration
│   │   ├── types.py              # Data types (ToolCall, ToolResult, BuyerDecision)
│   │   ├── config.py             # Configuration classes (SalesBenchConfig)
│   │   └── protocol.py           # Tool schemas and definitions
│   │
│   ├── agents/                   # AI agents (buyer and seller)
│   │   ├── buyer_llm.py          # LLM-based buyer simulator
│   │   ├── seller_base.py        # Base class for seller agents
│   │   └── seller_llm.py         # LLM-based seller agent
│   │
│   ├── llm/                      # Multi-provider LLM client
│   │   ├── __init__.py           # create_client(), detect_available_provider()
│   │   └── client.py             # LLMClient adapter
│   │
│   ├── envs/sales_mvp/           # Sales environment implementation
│   │   ├── env.py                # SalesEnv class (core simulation)
│   │   ├── state.py              # Environment state
│   │   ├── personas.py           # PersonaGenerator
│   │   ├── products.py           # ProductCatalog
│   │   │
│   │   ├── tools/                # Tool implementations
│   │   │   ├── crm.py            # CRM tools
│   │   │   ├── calendar.py       # Calendar tools
│   │   │   ├── calling.py        # Calling tools
│   │   │   └── products.py       # Product tools
│   │   │
│   │   └── verifiers/            # Scoring and verification
│   │       └── scoring.py        # Score calculation
│   │
│   ├── orchestrator/             # Episode orchestration
│   │   └── orchestrator.py       # Orchestrator class
│   │
│   └── cli/                      # Command-line interface
│       └── main.py               # CLI commands
│
├── tests/                        # Test suite
├── pyproject.toml                # Package configuration
├── envhub.yaml                   # Prime Intellect Environment Hub config
├── .env.example                  # Example environment variables
└── README.md                     # This file
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Prime Intellect Verifiers                          │
│                              vf.load_environment()                           │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SalesBenchToolEnv                                   │
│                   (environment.py - vf.StatefulToolEnv)                      │
│                                                                              │
│  • 12 async tool functions                                                   │
│  • Rubric with scoring functions                                             │
│  • Dataset generation for episodes                                           │
└──────────────┬──────────────────────────────────┬───────────────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────────┐    ┌──────────────────────────────────────────┐
│        Tool Functions        │    │              SalesEnv                     │
│    (environment.py)          │    │    (envs/sales_mvp/env.py)               │
│                              │    │                                          │
│  • crm_search_leads()        │    │  • Core simulation                       │
│  • calling_propose_plan()    │───▶│  • State management                      │
│  • products_quote_premium()  │    │  • Buyer simulator integration           │
└──────────────────────────────┘    └──────────────────────────────────────────┘
```

---

## License

MIT
