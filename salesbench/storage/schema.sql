-- SalesBench Supabase Schema
-- Run this in your Supabase SQL editor to create the required tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Episodes table: stores completed benchmark episodes
CREATE TABLE IF NOT EXISTS salesbench_episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id TEXT UNIQUE NOT NULL,
    seed INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    num_leads INTEGER DEFAULT 100,
    total_hours INTEGER DEFAULT 80,
    final_score DOUBLE PRECISION,
    metrics JSONB,
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying by model
CREATE INDEX IF NOT EXISTS idx_episodes_model ON salesbench_episodes(model_name);
CREATE INDEX IF NOT EXISTS idx_episodes_seed ON salesbench_episodes(seed);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON salesbench_episodes(started_at DESC);

-- Events table: stores all events from episodes
CREATE TABLE IF NOT EXISTS salesbench_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id TEXT NOT NULL,
    episode_id TEXT NOT NULL REFERENCES salesbench_episodes(episode_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying events
CREATE INDEX IF NOT EXISTS idx_events_episode ON salesbench_events(episode_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON salesbench_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON salesbench_events(timestamp);

-- Metrics table: stores aggregated metrics
CREATE TABLE IF NOT EXISTS salesbench_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_id TEXT NOT NULL,
    episode_id TEXT NOT NULL REFERENCES salesbench_episodes(episode_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    tags JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for querying metrics
CREATE INDEX IF NOT EXISTS idx_metrics_episode ON salesbench_metrics(episode_id);
CREATE INDEX IF NOT EXISTS idx_metrics_model ON salesbench_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON salesbench_metrics(metric_name);

-- Leads table: stores generated leads for debugging
CREATE TABLE IF NOT EXISTS salesbench_leads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lead_id TEXT NOT NULL,
    episode_id TEXT NOT NULL REFERENCES salesbench_episodes(episode_id) ON DELETE CASCADE,
    seed INTEGER NOT NULL,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    job TEXT,
    annual_income INTEGER,
    temperature TEXT,
    risk_class TEXT,
    hidden_state JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(episode_id, lead_id)
);

-- Index for querying leads
CREATE INDEX IF NOT EXISTS idx_leads_episode ON salesbench_leads(episode_id);
CREATE INDEX IF NOT EXISTS idx_leads_temperature ON salesbench_leads(temperature);

-- Calls table: stores call records
CREATE TABLE IF NOT EXISTS salesbench_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id TEXT NOT NULL,
    episode_id TEXT NOT NULL REFERENCES salesbench_episodes(episode_id) ON DELETE CASCADE,
    lead_id TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    duration_minutes INTEGER,
    outcome TEXT,
    offers_presented JSONB,
    buyer_responses JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(episode_id, call_id)
);

-- Index for querying calls
CREATE INDEX IF NOT EXISTS idx_calls_episode ON salesbench_calls(episode_id);
CREATE INDEX IF NOT EXISTS idx_calls_lead ON salesbench_calls(lead_id);
CREATE INDEX IF NOT EXISTS idx_calls_outcome ON salesbench_calls(outcome);

-- View for model comparison
CREATE OR REPLACE VIEW salesbench_model_summary AS
SELECT
    model_name,
    COUNT(*) as total_episodes,
    AVG(final_score) as avg_score,
    STDDEV(final_score) as stddev_score,
    MIN(final_score) as min_score,
    MAX(final_score) as max_score,
    AVG((metrics->>'accepted_offers')::int) as avg_accepts,
    AVG((metrics->>'rejected_offers')::int) as avg_rejects,
    AVG((metrics->>'dnc_violations')::int) as avg_dnc_violations
FROM salesbench_episodes
WHERE final_score IS NOT NULL
GROUP BY model_name
ORDER BY avg_score DESC;

-- View for per-seed reproducibility check
CREATE OR REPLACE VIEW salesbench_seed_variance AS
SELECT
    seed,
    model_name,
    COUNT(*) as runs,
    AVG(final_score) as avg_score,
    STDDEV(final_score) as score_variance
FROM salesbench_episodes
WHERE final_score IS NOT NULL
GROUP BY seed, model_name
HAVING COUNT(*) > 1
ORDER BY seed, model_name;

-- Row Level Security (optional - enable if needed)
-- ALTER TABLE salesbench_episodes ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE salesbench_events ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE salesbench_metrics ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE salesbench_leads ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE salesbench_calls ENABLE ROW LEVEL SECURITY;

-- Grant permissions (adjust as needed)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
