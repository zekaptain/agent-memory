-- Judge verdicts are machine decisions, not human actions.
INSERT INTO agents (agent_id, display_name, capture_tier, integration, notes) VALUES
    ('judge', 'Judge (mem judge)', 'A', 'cli', 'Experiment verdicts only; no lesson admission.');

-- Pending correction work is populated only by the ledger trigger.
CREATE OR REPLACE MACRO mem_authorize(sid, q) AS (
    regexp_full_match(mem_sql_normalize(q),
        ' ?(select|with|from|insert|begin|commit|rollback|abort|end|describe|show|explain|summarize)\b.*')
    AND NOT regexp_matches(mem_sql_normalize(q), '--|/\*|\$[a-z0-9_]*\$|\be''''')
    AND NOT regexp_matches(regexp_replace(mem_sql_normalize(q), '[; ]+$', ''), ';')
    AND NOT regexp_matches(mem_sql_normalize(q),
        '\b(update|delete|drop|alter|create|attach|detach|copy|export|import|install|load|pragma|set|reset|call|vacuum|checkpoint|force_checkpoint|truncate|merge|grant|revoke|use|connect|disconnect|prepare|execute|deallocate|query|query_table|enable_logging|disable_logging|truncate_duckdb_logs|enable_profiling|disable_profiling)\b')
    AND NOT regexp_matches(mem_sql_normalize(q), '\bquack_[a-z0-9_]*\s*\(')
    AND (NOT regexp_matches(mem_sql_normalize(q), '\binsert\b')
        OR regexp_full_match(mem_sql_normalize(q),
            ' ?insert into (main\.)?(sessions|events|memories|memory_sources|memory_actions|retrievals|retrieval_items|context_links|memory_deduplications|memory_replacements|memory_correction_closures|capture_observations|answer_bindings)\b.*'))
);
