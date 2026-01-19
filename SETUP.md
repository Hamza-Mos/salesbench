# SalesBench Production Setup Guide

This guide walks you through setting up SalesBench for production benchmarking with Supabase, Grafana/OpenTelemetry, and various LLM providers.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supabase Setup](#supabase-setup)
3. [Grafana + OpenTelemetry Setup](#grafana--opentelemetry-setup)
4. [LLM Provider Setup](#llm-provider-setup)
5. [Environment Variables](#environment-variables)
6. [Running Benchmarks](#running-benchmarks)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install with all dependencies
pip install -e ".[all]"

# 2. Copy and edit environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Test the setup (no integrations)
salesbench run-benchmark --mode debug --no-supabase --no-telemetry

# 4. Run with integrations
salesbench run-benchmark --mode test
```

---

## Supabase Setup

### Step 1: Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign in
2. Click **"New Project"**
3. Fill in:
   - **Name**: `salesbench` (or your preference)
   - **Database Password**: Generate a strong password and save it
   - **Region**: Choose closest to you
4. Click **"Create new project"** and wait for provisioning (~2 minutes)

### Step 2: Get Your API Keys

1. In your project dashboard, go to **Settings** → **API**
2. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon public key**: `eyJhbG...` (for read access)
   - **service_role key**: `eyJhbG...` (for write access - keep secret!)

### Step 3: Create the Database Tables

1. Go to **SQL Editor** in your Supabase dashboard
2. Click **"New query"**
3. Copy and paste the contents of `scripts/database/001_create_tables.sql`
4. Click **"Run"** (or Cmd+Enter)
5. You should see "Success. No rows returned"

### Step 4: Verify Tables Were Created

1. Go to **Table Editor** in the sidebar
2. You should see these tables:
   - `salesbench_episodes`
   - `salesbench_events`
   - `salesbench_metrics`
   - `salesbench_leads`
   - `salesbench_calls`
   - `salesbench_benchmarks`

### Step 5: Add to Environment

```bash
# In your .env file:
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## Grafana + OpenTelemetry Setup

You have two options: Grafana Cloud (easiest) or self-hosted.

### Option A: Grafana Cloud (Recommended)

#### Step 1: Create Grafana Cloud Account

1. Go to [grafana.com/products/cloud](https://grafana.com/products/cloud/)
2. Sign up for free tier (includes 10k traces/month)
3. Create a new stack or use default

#### Step 2: Get OTLP Endpoint

1. In Grafana Cloud, go to **Connections** → **Add new connection**
2. Search for **"OpenTelemetry (OTLP)"**
3. Click **"Configure"**
4. Note your OTLP endpoint: `https://otlp-gateway-prod-us-central-0.grafana.net/otlp`
5. Generate an API token with **MetricsPublisher** and **TracesPublisher** scopes

#### Step 3: Add to Environment

```bash
# In your .env file:
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-prod-us-central-0.grafana.net/otlp
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64-encoded-instance-id:api-token>

GRAFANA_URL=https://your-stack.grafana.net
GRAFANA_API_KEY=glc_xxxxx
```

### Option B: Self-Hosted (Docker)

#### Step 1: Create Docker Compose File

Create `docker-compose.otel.yml`:

```yaml
version: '3.8'

services:
  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
      - "8889:8889"   # Prometheus metrics

  # Jaeger for trace visualization
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14250:14250"  # gRPC

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  grafana-data:
```

#### Step 2: Create OTel Collector Config

Create `otel-collector-config.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: salesbench

  logging:
    verbosity: detailed

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger, logging]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus, logging]
```

#### Step 3: Create Prometheus Config

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']
```

#### Step 4: Start Services

```bash
docker-compose -f docker-compose.otel.yml up -d
```

#### Step 5: Configure Grafana

1. Open http://localhost:3000 (admin/admin)
2. Add Jaeger data source: http://jaeger:16686
3. Add Prometheus data source: http://prometheus:9090
4. Import dashboards or create your own

#### Step 6: Add to Environment

```bash
# In your .env file:
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
GRAFANA_URL=http://localhost:3000
```

---

## LLM Provider Setup

SalesBench supports multiple LLM providers. You need at least one configured.

### OpenAI

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click **"Create new secret key"**
3. Copy the key (starts with `sk-`)

```bash
OPENAI_API_KEY=sk-xxxxx
```

### Anthropic (Claude)

1. Go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
2. Click **"Create Key"**
3. Copy the key (starts with `sk-ant-`)

```bash
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

### OpenRouter (Access Multiple Models)

OpenRouter gives you access to many models through one API.

1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
2. Click **"Create Key"**
3. Add credits (pay-as-you-go)

```bash
OPENROUTER_API_KEY=sk-or-xxxxx
```

### Prime Intellect

Prime Intellect provides access to open-source models.

1. Go to [app.primeintellect.ai](https://app.primeintellect.ai/)
2. Sign up / Log in
3. Go to **API Keys** in your dashboard
4. Create a new API key

```bash
# Prime Intellect uses OpenAI-compatible API
# Set the base URL and API key:
PRIMEINTELLECT_API_KEY=pi-xxxxx
PRIMEINTELLECT_BASE_URL=https://api.primeintellect.ai/v1
```

To use Prime Intellect models, you'll need to add support to the LLM module. Here's how:

```python
# In your code or as environment variables:
SALESBENCH_SELLER_MODEL=deepseek-r1  # or other PI model
```

### Google (Gemini)

1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **"Create API key"**

```bash
GOOGLE_API_KEY=AIza...
```

### xAI (Grok)

1. Go to [console.x.ai](https://console.x.ai/)
2. Create an API key

```bash
XAI_API_KEY=xai-xxxxx
```

### Together AI

1. Go to [api.together.xyz/settings/api-keys](https://api.together.xyz/settings/api-keys)
2. Create an API key

```bash
TOGETHER_API_KEY=xxxxx
```

---

## Environment Variables

Create a `.env` file in your project root:

```bash
# =============================================================================
# LLM Providers (at least one required)
# =============================================================================
OPENAI_API_KEY=sk-xxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxxx
# OPENROUTER_API_KEY=sk-or-xxxxx
# GOOGLE_API_KEY=AIza...
# XAI_API_KEY=xai-xxxxx
# TOGETHER_API_KEY=xxxxx

# Prime Intellect (OpenAI-compatible)
# PRIMEINTELLECT_API_KEY=pi-xxxxx
# PRIMEINTELLECT_BASE_URL=https://api.primeintellect.ai/v1

# =============================================================================
# Model Configuration
# =============================================================================
# Override default models (optional)
SALESBENCH_SELLER_MODEL=gpt-4o
SALESBENCH_BUYER_MODEL=gpt-4o-mini
SALESBENCH_BUYER_TEMPERATURE=0.3

# =============================================================================
# Supabase (optional - for persistent storage)
# =============================================================================
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbG...
SUPABASE_SERVICE_ROLE_KEY=eyJhbG...

# =============================================================================
# OpenTelemetry (optional - for tracing/metrics)
# =============================================================================
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
# For Grafana Cloud:
# OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64>

# =============================================================================
# Grafana (optional - for trace links)
# =============================================================================
GRAFANA_URL=http://localhost:3000
# GRAFANA_API_KEY=glc_xxxxx
```

---

## Running Benchmarks

### Test Mode (Quick Validation)

```bash
# Run 3 episodes with 5 leads each
salesbench run-benchmark --mode test
```

### Production Mode

```bash
# Run 100 episodes with 100 leads each
salesbench run-benchmark --mode production --parallelism 10

# With specific model
salesbench run-benchmark --episodes 100 --seller-model gpt-4o --parallelism 10

# Export results
salesbench run-benchmark --mode production --output results.json
```

### Debug Mode (Single Episode)

```bash
# Verbose single episode
salesbench run-benchmark --mode debug --verbose
```

### Without Integrations

```bash
# Disable Supabase and telemetry
salesbench run-benchmark --mode test --no-supabase --no-telemetry
```

---

## Viewing Results

### In Supabase

1. Go to your Supabase dashboard → **Table Editor**
2. Select `salesbench_episodes` to see episode results
3. Select `salesbench_metrics` to see detailed metrics
4. Use **SQL Editor** with queries from `scripts/database/002_sample_queries.sql`

### In Grafana

1. Open your Grafana instance
2. Go to **Explore**
3. Select Jaeger/Tempo data source
4. Search for traces by `service.name = salesbench`
5. Filter by `episode.id` or `benchmark.id`

### In JSON Output

```bash
# Run with output
salesbench run-benchmark --mode test --output results.json

# View results
cat results.json | jq '.aggregate_metrics'
```

---

## Troubleshooting

### "No LLM provider API key found"

Make sure at least one provider key is set:
```bash
echo $OPENAI_API_KEY  # Should show your key
```

### "Supabase not configured"

Check your Supabase credentials:
```bash
echo $SUPABASE_URL  # Should show your URL
```

### "Telemetry disabled"

Telemetry requires `OTEL_ENABLED=true` and valid endpoint:
```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### "Failed to flush to table"

1. Check your Supabase service role key has write permissions
2. Verify tables exist (run the SQL migration)
3. Check for network connectivity

### "Rate limit exceeded"

Reduce parallelism:
```bash
salesbench run-benchmark --parallelism 2
```

### Install Missing Packages

```bash
# Install all optional dependencies
pip install -e ".[all]"

# Or specific ones
pip install -e ".[llm,telemetry,storage]"
```

---

## Architecture Overview

```
┌─────────────────┐
│  CLI Command    │
│ run-benchmark   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BenchmarkRunner │
│  (parallel)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Episode│ │Episode│ ...
│Exec 1 │ │Exec 2 │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐     ┌─────────────┐
│ IntegrationMgr  │────▶│  Supabase   │
│                 │     └─────────────┘
│                 │     ┌─────────────┐
│                 │────▶│  Grafana/   │
│                 │     │  OTel       │
└─────────────────┘     └─────────────┘
```

---

## Prime Intellect Integration

SalesBench is designed to be compatible with Prime Intellect's Verifiers and Environment Hub.

### Environment Interface

The benchmark exposes a standard Gym-like interface:

```python
from salesbench import load_environment, ToolCall

# Load environment
env = load_environment(seed=42, num_leads=100)

# Reset and get initial observation
obs = env.reset()

# Run episode
while not env.is_done:
    # Generate tool calls (from your agent)
    tool_calls = [
        ToolCall(tool_name="crm.search_leads", arguments={}),
    ]

    # Step environment
    obs, reward, done, info = env.step(tool_calls)

    if done:
        print(f"Final score: {info['score_breakdown']['total_score']}")
```

### Verifier Server

Run the verifier as an HTTP service for Prime Intellect integration:

```bash
# Install verifier dependencies
pip install -e ".[verifier]"

# Start verifier server
python -m salesbench.envs.sales_mvp.verifiers.server --port 8000

# Or with uvicorn
uvicorn salesbench.envs.sales_mvp.verifiers.server:get_app --port 8000
```

Endpoints:
- `GET /health` - Health check
- `GET /info` - Verifier metadata
- `POST /verify` - Verify episode trajectory
- `POST /score` - Score final state

### Environment Hub Registration

The `envhub.yaml` file contains the environment manifest for Env Hub:

```yaml
name: salesbench
version: 0.1.0
environment:
  entry_point: salesbench:load_environment

scoring:
  primary_metric: total_score
  pass_threshold: 0

verifier:
  entry_point: salesbench.envs.sales_mvp.verifiers:calculate_episode_score
```

### Scoring Rubric

The scoring system is RL-ready with clear reward signals:

| Event | Score |
|-------|-------|
| Accept plan | +100 |
| Close now bonus | +50 |
| Premium bonus | +1 × monthly_premium |
| Reject plan | -5 |
| End call | -10 |
| DNC violation | -200 |

### Using with Prime Intellect API

```python
# Using Prime Intellect as the LLM provider
export PRIMEINTELLECT_API_KEY=pi-xxxxx

# Run benchmark with DeepSeek
salesbench run-benchmark --mode test --seller-model deepseek-ai/DeepSeek-R1
```

### Submitting to Env Hub

1. Ensure all tests pass: `pytest`
2. Verify environment loads: `python -c "from salesbench import load_environment; env = load_environment(); print('OK')"`
3. Submit via Prime Intellect CLI or web interface

---

## Next Steps

1. **Set up Supabase** for persistent storage
2. **Configure at least one LLM provider**
3. **Run a test benchmark** to validate setup
4. **Optionally set up Grafana** for visualization
5. **Run production benchmarks** and analyze results
6. **Deploy verifier server** for Prime Intellect integration

For questions or issues, open a GitHub issue or check the documentation.
