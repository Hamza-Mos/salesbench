-- =============================================================================
-- SalesBench Sample Queries
-- Useful queries for analyzing benchmark results
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Basic Queries
-- -----------------------------------------------------------------------------

-- Get all episodes for a specific benchmark
SELECT * FROM salesbench_episode_summary
WHERE benchmark_id = 'bench_abc12345';

-- Get recent episodes
SELECT * FROM salesbench_episode_summary
LIMIT 50;

-- Get model performance comparison
SELECT * FROM salesbench_model_performance;

-- -----------------------------------------------------------------------------
-- Metrics Analysis
-- -----------------------------------------------------------------------------

-- Get all metrics for an episode
SELECT metric_name, metric_value, timestamp
FROM salesbench_metrics
WHERE episode_id = 'bench_abc_ep0_123456'
ORDER BY metric_name;

-- Get aggregate metrics for a benchmark
SELECT metric_name, AVG(metric_value) as avg_value, STDDEV(metric_value) as std_value
FROM salesbench_metrics
WHERE tags->>'benchmark_id' = 'bench_abc12345'
  AND tags->>'type' != 'aggregate'
GROUP BY metric_name
ORDER BY metric_name;

-- Compare models by metric
SELECT
    model_name,
    AVG(CASE WHEN metric_name = 'final_score' THEN metric_value END) as avg_score,
    AVG(CASE WHEN metric_name = 'total_accepts' THEN metric_value END) as avg_accepts,
    AVG(CASE WHEN metric_name = 'acceptance_rate' THEN metric_value END) as avg_acceptance_rate
FROM salesbench_metrics
WHERE metric_name IN ('final_score', 'total_accepts', 'acceptance_rate')
GROUP BY model_name
ORDER BY avg_score DESC;

-- -----------------------------------------------------------------------------
-- Time Series Analysis
-- -----------------------------------------------------------------------------

-- Score over time for a model
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    model_name,
    AVG(metric_value) as avg_score,
    COUNT(*) as episodes
FROM salesbench_metrics
WHERE metric_name = 'final_score'
GROUP BY DATE_TRUNC('hour', timestamp), model_name
ORDER BY hour DESC;

-- Daily benchmark summary
SELECT
    DATE(started_at) as date,
    model_name,
    COUNT(*) as episodes,
    AVG(final_score) as avg_score,
    SUM((metrics->>'accepted_offers')::int) as total_accepts
FROM salesbench_episodes
WHERE final_score IS NOT NULL
GROUP BY DATE(started_at), model_name
ORDER BY date DESC, avg_score DESC;

-- -----------------------------------------------------------------------------
-- Episode Success Rate Analysis
-- -----------------------------------------------------------------------------

-- Calculate episode success rate for a benchmark
WITH episode_success AS (
    SELECT
        episode_id,
        (metrics->>'accepted_offers')::int > 0 as has_accepts
    FROM salesbench_episodes
    WHERE config->>'benchmark_id' = 'bench_abc12345'
)
SELECT
    COUNT(*) as total_episodes,
    SUM(CASE WHEN has_accepts THEN 1 ELSE 0 END) as successful_episodes,
    SUM(CASE WHEN has_accepts THEN 1 ELSE 0 END)::float / COUNT(*) as episode_success_rate
FROM episode_success;

-- -----------------------------------------------------------------------------
-- Cleanup Queries (Use with caution!)
-- -----------------------------------------------------------------------------

-- Delete old test data (older than 7 days)
-- DELETE FROM salesbench_metrics WHERE timestamp < NOW() - INTERVAL '7 days';
-- DELETE FROM salesbench_events WHERE timestamp < NOW() - INTERVAL '7 days';
-- DELETE FROM salesbench_episodes WHERE started_at < NOW() - INTERVAL '7 days';

-- Delete a specific benchmark's data
-- DELETE FROM salesbench_metrics WHERE tags->>'benchmark_id' = 'bench_to_delete';
-- DELETE FROM salesbench_episodes WHERE config->>'benchmark_id' = 'bench_to_delete';
