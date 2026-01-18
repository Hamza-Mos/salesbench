---
name: SalesBench Architecture (Prime Intellect compatible)
overview: A Prime Intellect Verifiers multi-turn + tool-use environment for life insurance cold-calling with an Orchestrator (state machine), canonical Environment state + tools, a Buyer simulator that only returns `{ACCEPT_PLAN|REJECT_PLAN|END_CALL}` inside `calling.propose_plan`, seeded persona generation, and an RL-ready rubric. Optional Supabase/Grafana integrations live outside the publishable env package.
todos:
  - id: publishable-env
    content: Create a publishable Verifiers env package under `environments/salesbench/` exporting `load_environment(...) -> vf.Environment` with Hub-safe dependencies.
    status: pending
  - id: core-types-protocol
    content: Implement core schemas (ToolCall/ToolResult/Decision/IDs) and strict protocol validators (seller tool-only MVP; buyer decision-only; buyer invoked only via calling tools).
    status: pending
  - id: personas-products
    content: Implement seeded persona generation (public persona + hidden state + random events) and define the canonical life insurance products/plans to sell (pricing + eligibility).
    status: pending
  - id: env-state-tools
    content: Implement canonical environment state (Leads/CRM/Calendar/CallSessions/EventQueue) and minimal tools (CRM, Calendar, Calling) executed only inside the Environment.
    status: pending
  - id: orchestrator
    content: Implement Orchestrator as environment transition logic (10 business days, budgets, termination) that calls the seller agent/model and routes tool calls to Environment.execute_tool.
    status: pending
  - id: buyer-sim
    content: Implement Buyer LLM simulator invoked only by `calling.propose_plan`, constrained to `{ACCEPT_PLAN|REJECT_PLAN|END_CALL}` + optional reason string.
    status: pending
  - id: scoring-rubric
    content: Implement Verifiers rubric reward + metrics (close/accept rate, profit proxy, time efficiency, inference cost) and pass^k aggregation hooks.
    status: pending
  - id: optional-supabase-otel
    content: Optional outer-layer runner integrations for Supabase batch writes and OpenTelemetry→Grafana (kept out of the publishable env package).
    status: pending
---

## Goals

- **Prime Intellect compatible**: publishable Verifiers environment with `load_environment(...) -> vf.Environment`.
- **Benchmark-aligned**: 100 leads, 10 simulated business days, multi-call memory, realistic interruptions.
- **Strict protocol**: seller uses tools; buyer emits only `{ACCEPT_PLAN|REJECT_PLAN|END_CALL}` during calls.
- **Environment is canonical**: state and tools live in the environment; orchestrator is a state machine (not an agent).
- **RL-ready**: bounded horizons, serializable state, rubric reward + metrics suitable for later Prime-RL.

## Architecture

### High-level components

- **Runner (CLI/API, optional outer layer)**: starts N episodes (seeded), sets configs/parallelism, and (optionally) streams telemetry/storage.
- **Orchestrator**: a turn/state machine that owns time (10 business days), budgets, and termination; it calls the seller and routes tool calls.
- **Seller (evaluated model)**: proposes actions via tool calls (MVP).
- **Environment**: canonical state owner (Leads/CRM/Calendar/CallSessions/EventQueue) and the only place tools execute.
- **Buyer simulator**: activated only inside `calling.propose_plan`, returning only a Decision (plus an optional reason string for the tool result).
- **ContextManager**: shared context builder/compactor for seller + buyer LLM calls.

### Core loop (compatible with Verifiers + later Prime-RL)

```mermaid
sequenceDiagram
participant VF as VF
participant Env as SalesBenchEnv
participant Orch as Orchestrator
participant Seller as SellerModel
participant Tools as EnvTools
participant Buyer as BuyerSim

VF->>Env: load_environment(...)
VF->>Env: reset(example_seed)
loop multi_turn
  VF->>Seller: generate(messages, tools)
  Seller-->>VF: tool_calls
  VF->>Env: step(tool_calls)
  Env->>Orch: advance(time/budgets)
  Env->>Tools: execute_tool(...)
  alt calling.propose_plan
    Tools->>Buyer: generate_next(call_context)
    Buyer-->>Tools: Decision
  end
  Env-->>VF: observation/messages + tool_results
end
```

## Protocol (single source of truth)

Implement in `salesbench/core/protocol.py`:

- **Seller (MVP)**: tool calls only (no free-form dialogue).
- **Buyer**: decision only (`ACCEPT_PLAN | REJECT_PLAN | END_CALL`), never calls tools.
- **Buyer activation**: buyer decisions occur only via the return of `calling.propose_plan(...)`.
- **Environment-only tools**: all tool side effects happen inside `SalesEnv.execute_tool(...)`.

## Tools (minimal MVP)

### Tool access (explicit)

- **Seller tool access**: may call `crm.*`, `calendar.*`, and `calling.*` tools (subject to budgets/termination enforced by the Orchestrator and Environment).
- **Buyer tool access**: none. The buyer never issues tool calls; it is invoked internally by the `calling.propose_plan` tool and can only return a Decision.
- **Orchestrator tool access**: none directly. It routes seller tool calls to `SalesEnv.execute_tool(...)` and advances time/budgets/termination.
- **Environment tool access**: executes all tools and applies state transitions; it is the only canonical owner of side effects.

### CRM

- `crm.search_leads(filters)`
- `crm.get_lead(lead_id)`
- `crm.update_lead(lead_id, patch)`
- `crm.log_call(lead_id, timestamp, outcome, plan_summary)`

### Calendar

- `calendar.get_availability(day)`
- `calendar.schedule_call(lead_id, datetime)`

### Calling (Buyer only “speaks” here)

- `calling.start_call(lead_id) -> call_id`
- `calling.propose_plan(call_id, plan) -> {decision: ACCEPT|REJECT|END, reason}`
- `calling.end_call(call_id, reason)`

## Persona generation (seeded, realistic)

Each lead has a deterministic persona generated from `(episode_seed, lead_id)`.

### Persona dimensions (MVP)

- **Public fields**: name, age, job category, income band, household, “trigger” (house, baby, health scare), objection style, timezone/availability hints.
- **Hidden state**: `trust`, `interest`, `patience`, `dnc_risk`, `close_threshold`, plus optional “price_sensitivity” and “time_sensitivity”.
- **Multi-call memory**: remembers promises/claims, previous plan summaries, and bad experiences; this influences future decisions.

### Archetype-driven sampling

Define ~10–20 archetypes with parameter ranges, e.g.:

- analytical_lukewarm (low initial trust, medium interest, high price sensitivity)
- hostile_cold (very low trust, high dnc_risk)
- warm_but_busy (high interest, low patience, time crunch prone)
- skeptical_budget (medium trust, low budget, asks for numbers)

### Random events (seeded)

Events sampled per call (with fixed probabilities, seeded):

- spouse_involvement
- child_interruption
- competing_priority
- emotional_trigger

Events modify hidden state and impose call constraints (e.g., “wrap up in 2 minutes”).

## Life insurance products/plans (canonical catalog)

Define a small, fixed product catalog so recommendations and outcomes are comparable:

### Products (MVP)

- **Term Life 10-year**: low premium, short horizon.
- **Term Life 20-year**: common family/mortgage fit.
- **Whole Life**: higher premium, legacy/cash-value framing.
- **Universal Life (simplified)**: flexible premiums; optional if we want diversity.

### Coverage tiers (MVP)

- \(250k\), \(500k\), \(1M\) face amount tiers.

### Riders (optional toggles)

- accidental_death
- child_rider
- waiver_of_premium

### Pricing + eligibility (deterministic, tunable)

- Monthly premium computed from age band, risk band, product type multiplier, coverage tier.
- Underwriting mode (simplified vs fully underwritten) affects acceptance probability and time-to-close.

The seller’s plan submitted to `calling.propose_plan` should reference a product + coverage tier + next step (schedule follow-up, gather info, etc.).

## Scoring / rubric (RL-ready)

Implement in `salesbench/envs/sales_mvp/verifiers/scoring.py`:

- **Primary reward (terminal)**: weighted combination of
  - accepts / closes (or scheduled qualified follow-up)
  - profit proxy (premium * retention proxy)
  - time efficiency (calls/day, wasted calls, over-calling penalties)
  - inference cost (tokens / $ cost from model requests)
- **Metrics (non-reward)**: accepts, rejects, ends, DNC events, avg call length, follow-up rate, cost, etc.

Ensure bounded horizons:

- max calls/day
- max tool steps/episode
- max turns/call session

## Directory structure

```text
salesbench/
  environments/
    salesbench/
      salesbench.py            # Verifiers entrypoint: load_environment(...)
      pyproject.toml           # env package metadata + [tool.verifiers.eval] defaults
      README.md

  salesbench/
    core/
      types.py
      protocol.py
      config.py
      errors.py

    context/
      manager.py
      buffers.py
      compaction/
        base.py
        simple_summary.py
        key_events.py
      policies.py
      serializers.py

    agents/
      base.py
      seller_llm.py
      seller_heuristic.py
      buyer_llm.py

    orchestrator/
      orchestrator.py
      budgets.py
      termination.py

    envs/
      sales_mvp/
        env.py
        state.py
        personas.py
        products.py
        tools/
          crm.py
          calendar.py
          calling.py
          tool_metadata.yaml
        verifiers/
          scoring.py

    llm/
      client.py
      tool_schema.py
      parsing.py

    storage/
      supabase_writer.py        # optional outer-layer integration
      schema.sql

    telemetry/
      otel.py                   # optional outer-layer integration
      spans.py

    cli/
      main.py                   # optional runner entrypoint
```

## Implementation plan (MVP-first)

1. **Publishable Verifiers environment** (`environments/salesbench/`): implement `load_environment(...)` and default eval config.
2. **Core types + protocol**: strict seller/buyer constraints in `core/protocol.py`.
3. **Personas + products**: seeded persona generator + product catalog/pricing in `envs/sales_mvp/`.
4. **Environment + tools**: canonical state + CRM/Calendar/Calling tools; buyer invoked only inside calling tool.
5. **Orchestrator**: 10-day portfolio loop, budgets/termination, call session flow.
6. **Rubric**: reward + metrics; pass^k aggregation hooks.
7. **Baselines**: heuristic seller + LLM seller; constrained buyer LLM sim.
8. **Optional ops layer**: Supabase + OTel/Grafana integrations outside the publishable env package.