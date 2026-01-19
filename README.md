# SalesBench

AI Social Intelligence Benchmark for evaluating agents through life insurance cold-calling scenarios. Built for Prime Intellect Verifiers.

## Features

- **LLM-based buyer simulation** - Realistic buyer behavior using any LLM provider
- **Multi-provider support** - OpenAI, Anthropic, OpenRouter, xAI (Grok), Together AI, Google Gemini
- **Seeded persona generation** - 100 reproducible leads per episode
- **12 seller tools** - CRM, Calendar, Calling, Products
- **RL-ready scoring** - Bounded rewards, serializable state
- **Prime Intellect Verifiers compatible**

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
python -m salesbench seed-leads --seed 42 --n 5
python -m salesbench inspect-products

# This needs an API key:
python -m salesbench run-episode --seed 42 --leads 5 --max-turns 10 -v
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
from salesbench.agents import LLMSellerAgent

# Auto-detect provider
env = load_environment(seed=42)

# Or specify provider explicitly
from salesbench.llm import create_client
client = create_client(provider="anthropic", model="claude-3-5-sonnet-20241022")
```

### Using OpenRouter for Any Model

OpenRouter gives you access to 100+ models through a single API:

```bash
OPENROUTER_API_KEY=sk-or-your-key
```

```python
from salesbench.llm import create_client

# Use any model via OpenRouter
client = create_client(
    provider="openrouter",
    model="anthropic/claude-3.5-sonnet",  # or any OpenRouter model
)
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

### Database (Optional - Not needed for testing)

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anonymous key |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |

### Monitoring (Optional - Not needed for testing)

| Variable | Description |
|----------|-------------|
| `GRAFANA_URL` | Grafana URL (default: `http://localhost:3000`) |
| `GRAFANA_API_KEY` | Grafana API key |
| `PROMETHEUS_PUSH_GATEWAY` | Prometheus push gateway URL |

---

## Running the Benchmark Seriously

### Basic Benchmark Run

```python
import os
from salesbench import load_environment
from salesbench.agents import LLMSellerAgent

# Set your API key
os.environ["OPENAI_API_KEY"] = "sk-your-key"

# Create environment
env = load_environment(seed=42, num_leads=100)

# Create seller agent
seller = LLMSellerAgent()  # Auto-detects provider

# Run episode
obs = env.reset()
total_reward = 0

while not env.is_done:
    tool_calls = seller.act(obs)
    obs, reward, done, info = env.step(tool_calls)
    total_reward += reward

print(f"Total reward: {total_reward}")
print(f"Metrics: {info.get('metrics', {})}")
```

### Multi-Seed Evaluation (For Statistical Significance)

```python
import json
from salesbench import load_environment
from salesbench.agents import LLMSellerAgent

def evaluate_model(provider: str, model: str, seeds: list[int] = range(10)):
    """Evaluate a model across multiple seeds."""
    results = []

    for seed in seeds:
        env = load_environment(seed=seed, num_leads=100)
        seller = LLMSellerAgent(provider=provider, model=model)

        obs = env.reset()
        total_reward = 0

        while not env.is_done:
            tool_calls = seller.act(obs)
            obs, reward, done, info = env.step(tool_calls)
            total_reward += reward

        metrics = info.get("metrics", {})
        results.append({
            "seed": seed,
            "reward": total_reward,
            "accepts": metrics.get("accepted_offers", 0),
            "rejects": metrics.get("rejected_offers", 0),
            "calls": metrics.get("total_calls", 0),
        })

        print(f"Seed {seed}: reward={total_reward:.1f}, accepts={results[-1]['accepts']}")

    # Compute statistics
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_accepts = sum(r["accepts"] for r in results) / len(results)

    return {
        "provider": provider,
        "model": model,
        "num_seeds": len(seeds),
        "avg_reward": avg_reward,
        "avg_accepts": avg_accepts,
        "results": results,
    }

# Compare models
gpt4_results = evaluate_model("openai", "gpt-4o")
gpt4mini_results = evaluate_model("openai", "gpt-4o-mini")
claude_results = evaluate_model("anthropic", "claude-3-5-sonnet-20241022")

# Save results
with open("benchmark_results.json", "w") as f:
    json.dump([gpt4_results, gpt4mini_results, claude_results], f, indent=2)
```

### Pass@K Evaluation

```python
from salesbench.metrics import PassAtKComputer

# Compute pass@k where "pass" = at least 5 accepts
computer = PassAtKComputer(
    n_samples=100,
    k_values=[1, 5, 10],
    pass_threshold=5,  # At least 5 accepts to "pass"
)

# Run 100 episodes
for seed in range(100):
    env = load_environment(seed=seed)
    seller = LLMSellerAgent()

    obs = env.reset()
    while not env.is_done:
        obs, _, _, info = env.step(seller.act(obs))

    accepts = info.get("metrics", {}).get("accepted_offers", 0)
    computer.add_result(accepts >= 5)

print(computer.compute())
# Output: {"pass@1": 0.23, "pass@5": 0.67, "pass@10": 0.89}
```

---

## CLI Commands

| Command | API Key? | Purpose |
|---------|----------|---------|
| `seed-leads` | No | Generate sample personas |
| `inspect-products` | No | Show insurance products |
| `quote` | No | Calculate premium |
| `run-episode` | **Yes** | Run full episode with LLM agents |

```bash
# Preview personas
python -m salesbench seed-leads --seed 42 --n 10

# See products
python -m salesbench inspect-products

# Calculate a quote
python -m salesbench quote --plan TERM --age 35 --coverage 500000 --term 20

# Run episode (requires API key)
python -m salesbench run-episode --seed 42 --leads 20 --max-turns 50 -v
```

---

## Available Tools

### CRM Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `crm.search_leads` | `temperature`, `min_income`, `max_age`, `limit` | Search leads |
| `crm.get_lead` | `lead_id` | Get lead details |
| `crm.update_lead` | `lead_id`, `notes`, `temperature` | Update lead |
| `crm.log_call` | `lead_id`, `call_id`, `outcome`, `notes` | Log call |

### Calendar Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `calendar.get_availability` | `day` | Get available slots |
| `calendar.schedule_call` | `lead_id`, `day`, `hour` | Schedule call |

### Calling Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `calling.start_call` | `lead_id` | Start call |
| `calling.propose_plan` | `plan_id`, `monthly_premium`, `coverage_amount`, `next_step` | Present offer |
| `calling.end_call` | `reason` | End call |

### Product Tools
| Tool | Arguments | Description |
|------|-----------|-------------|
| `products.list_plans` | - | List all plans |
| `products.get_plan` | `plan_id` | Get plan details |
| `products.quote_premium` | `plan_id`, `age`, `coverage_amount`, `risk_class`, `term_years` | Calculate premium |

---

## Scoring

| Event | Points | Notes |
|-------|--------|-------|
| Plan accepted | +100 | Base reward |
| Close now bonus | +50 | If `next_step=close_now` |
| Schedule followup bonus | +20 | If `next_step=schedule_followup` |
| Premium multiplier | +0.5 × premium | Higher premiums = more reward |
| Plan rejected | -5 | Per rejection |
| Buyer ends call | -10 | Buyer hangs up |
| DNC violation | -200 | Calling someone on Do Not Call list |

---

## Project Structure

```
salesbench/
├── environments/salesbench/
│   ├── __init__.py           # Main exports
│   ├── environment.py        # load_environment()
│   ├── core/                 # Types, config, errors
│   ├── llm/                  # Multi-provider LLM client
│   ├── envs/sales_mvp/       # Environment implementation
│   ├── agents/               # Buyer + Seller agents
│   ├── orchestrator/         # Episode management
│   └── cli/                  # CLI commands
├── pyproject.toml
├── .env.example
└── README.md
```

---

## License

MIT
