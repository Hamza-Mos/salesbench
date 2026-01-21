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

| Mode | Episodes | Leads | Days | Max Turns | Purpose |
|------|----------|-------|------|-----------|---------|
| `debug` | 1 | 5 | 10 | 50 | Quick sanity check |
| `test` | 3 | 5 | 10 | 30 | Fast validation |
| `production` | 100 | 100 | 10 | 200 | Official benchmark |

An episode ends when: (1) 10 simulated days elapse, (2) max turns reached, or (3) all leads are on DNC list.

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
salesbench run-benchmark   # Run benchmark
salesbench run-episode     # Single episode (debug)
salesbench list-models     # Show known models
salesbench list-domains    # Show available domains
salesbench leaderboard     # Launch web UI
salesbench seed-leads      # Preview personas
salesbench inspect-products # View products
```

## Verifiers Integration

```python
import verifiers as vf

env = vf.load_environment("salesbench", seed=42, num_leads=100)
results = env.evaluate(client, model="gpt-4o")
```

## License

MIT
