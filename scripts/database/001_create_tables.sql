-- SalesBench Database Schema
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard/project/YOUR_PROJECT/sql

-- =============================================================================
-- Episodes Table
-- Stores metadata for each episode run
-- =============================================================================
CREATE TABLE IF NOT EXISTS salesbench_episodes (
    id BIGSERIAL PRIMARY KEY,
    episode_id TEXT UNIQUE NOT NULL,
    seed INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    num_leads INTEGER DEFAULT 100,
    total_days INTEGER DEFAULT 10,
    final_score DECIMAL(12, 4),
    metrics JSONB,
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying by model
CREATE INDEX IF NOT EXISTS idx_episodes_model ON salesbench_episodes(model_name);
CREATE INDEX IF NOT EXISTS idx_episodes_seed ON salesbench_episodes(seed);
CREATE INDEX IF NOT EXISTS idx_episodes_started_at ON salesbench_episodes(started_at DESC);

-- =============================================================================
-- Events Table
-- Stores individual events during episodes (tool calls, decisions, etc.)
-- =============================================================================
CREATE TABLE IF NOT EXISTS salesbench_events (
    id BIGSERIAL PRIMARY KEY,
    event_id TEXT UNIQUE NOT NULL,
    episode_id TEXT NOT NULL REFERENCES salesbench_episodes(episode_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying events by episode
CREATE INDEX IF NOT EXISTS idx_events_episode ON salesbench_events(episode_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON salesbench_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON salesbench_events(timestamp DESC);

-- =============================================================================
-- Metrics Table
-- Stores individual metric values for analysis
-- =============================================================================
CREATE TABLE IF NOT EXISTS salesbench_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_id TEXT UNIQUE NOT NULL,
    episode_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DECIMAL(16, 6) NOT NULL,
    tags JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for metric queries
CREATE INDEX IF NOT EXISTS idx_metrics_episode ON salesbench_metrics(episode_id);
CREATE INDEX IF NOT EXISTS idx_metrics_model ON salesbench_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON salesbench_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON salesbench_metrics(timestamp DESC);

-- Composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_metrics_model_name ON salesbench_metrics(model_name, metric_name);

-- =============================================================================
-- Leads Table (Optional - for debugging/analysis)
-- Stores lead data for episodes
-- =============================================================================
CREATE TABLE IF NOT EXISTS salesbench_leads (
    id BIGSERIAL PRIMARY KEY,
    lead_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    name TEXT,
    age INTEGER,
    temperature TEXT,
    annual_income DECIMAL(12, 2),
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_leads_episode ON salesbench_leads(episode_id);

-- =============================================================================
-- Calls Table (Optional - for call-level analysis)
-- Stores individual call records
-- =============================================================================
CREATE TABLE IF NOT EXISTS salesbench_calls (
    id BIGSERIAL PRIMARY KEY,
    call_id TEXT UNIQUE NOT NULL,
    episode_id TEXT NOT NULL,
    lead_id TEXT NOT NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    duration_minutes INTEGER,
    outcome TEXT,
    offers_presented INTEGER DEFAULT 0,
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calls_episode ON salesbench_calls(episode_id);
CREATE INDEX IF NOT EXISTS idx_calls_lead ON salesbench_calls(lead_id);
CREATE INDEX IF NOT EXISTS idx_calls_outcome ON salesbench_calls(outcome);

-- =============================================================================
-- Benchmarks Table
-- Stores benchmark run summaries
-- =============================================================================
CREATE TABLE IF NOT EXISTS salesbench_benchmarks (
    id BIGSERIAL PRIMARY KEY,
    benchmark_id TEXT UNIQUE NOT NULL,
    name TEXT,
    mode TEXT NOT NULL,
    config JSONB,
    total_episodes INTEGER DEFAULT 0,
    completed_episodes INTEGER DEFAULT 0,
    failed_episodes INTEGER DEFAULT 0,
    aggregate_metrics JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    duration_seconds DECIMAL(12, 3),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_mode ON salesbench_benchmarks(mode);
CREATE INDEX IF NOT EXISTS idx_benchmarks_started_at ON salesbench_benchmarks(started_at DESC);

-- =============================================================================
-- Updated At Trigger
-- Automatically update the updated_at column
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_episodes_updated_at
    BEFORE UPDATE ON salesbench_episodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Useful Views
-- =============================================================================

-- View: Episode summary with key metrics
CREATE OR REPLACE VIEW salesbench_episode_summary AS
SELECT
    e.episode_id,
    e.seed,
    e.model_name,
    e.started_at,
    e.final_score,
    (e.metrics->>'accepted_offers')::int as accepts,
    (e.metrics->>'rejected_offers')::int as rejects,
    (e.metrics->>'total_calls')::int as total_calls,
    (e.metrics->>'dnc_violations')::int as dnc_violations,
    (e.config->>'benchmark_id')::text as benchmark_id
FROM salesbench_episodes e
ORDER BY e.started_at DESC;

-- View: Model performance comparison
CREATE OR REPLACE VIEW salesbench_model_performance AS
SELECT
    model_name,
    COUNT(*) as total_episodes,
    AVG(final_score) as avg_score,
    STDDEV(final_score) as std_score,
    AVG((metrics->>'accepted_offers')::int) as avg_accepts,
    SUM((metrics->>'accepted_offers')::int) as total_accepts,
    AVG((metrics->>'dnc_violations')::int) as avg_dnc_violations
FROM salesbench_episodes
WHERE final_score IS NOT NULL
GROUP BY model_name
ORDER BY avg_score DESC;

-- =============================================================================
-- Row Level Security (RLS) - Optional
-- Uncomment if you want to restrict access
-- =============================================================================

-- ALTER TABLE salesbench_episodes ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE salesbench_events ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE salesbench_metrics ENABLE ROW LEVEL SECURITY;

-- Allow service role full access
-- CREATE POLICY "Service role access" ON salesbench_episodes
--     FOR ALL USING (auth.role() = 'service_role');
