-- Coverage is observed, not inferred from agent capability or an immutable session row.
CREATE TABLE capture_observations (
    observation_id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions (session_id),
    source_mode VARCHAR NOT NULL CHECK (source_mode IN
        ('mcp', 'hook', 'record', 'transcript_import', 'supported_capture')),
    adapter_version VARCHAR NOT NULL,
    policy_version VARCHAR NOT NULL,
    source_fingerprint VARCHAR NOT NULL,
    covered_from TIMESTAMPTZ,
    covered_through TIMESTAMPTZ,
    cursor_start VARCHAR,
    cursor_end VARCHAR,
    range_start BIGINT,
    range_end BIGINT,
    parse_errors VARIANT NOT NULL,
    exclusions VARIANT NOT NULL,
    complete_supported BOOLEAN NOT NULL DEFAULT false,
    session_metadata VARIANT,
    attribution_evidence VARIANT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (covered_from IS NULL OR covered_through IS NULL OR covered_from <= covered_through),
    CHECK (range_start IS NULL OR range_end IS NULL OR range_start <= range_end),
    CHECK (NOT complete_supported OR
        (source_mode = 'supported_capture' AND covered_from IS NOT NULL
         AND covered_through IS NOT NULL
         AND json_array_length(parse_errors::JSON) = 0
         AND json_array_length(exclusions::JSON) = 0))
);
CREATE INDEX capture_observations_session ON capture_observations (session_id);
CREATE TRIGGER capture_observations_no_update AFTER UPDATE ON capture_observations REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('capture_observations is append-only') FROM o;
CREATE TRIGGER capture_observations_no_delete AFTER DELETE ON capture_observations REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('capture_observations is append-only') FROM o;

CREATE OR REPLACE VIEW session_summary AS
WITH counts AS (
    SELECT session_id,
        max(ts) FILTER (WHERE kind = 'session_end') AS ended_at,
        max(ts) AS last_event_at,
        count(*) AS event_count,
        count(*) FILTER (WHERE kind = 'assistant_msg'
            AND NOT coalesce(try_cast(payload.placeholder AS BOOLEAN), false)
            AND coalesce(payload.part::VARCHAR, '') <> 'thinking') AS answers,
        count(*) FILTER (WHERE kind = 'memory_recall') AS recalls,
        count(*) FILTER (WHERE kind = 'compaction') AS compactions,
        count(*) FILTER (WHERE kind = 'capture_error') AS capture_errors
    FROM events GROUP BY session_id
), observed AS (
    SELECT *, row_number() OVER (
        PARTITION BY session_id
        ORDER BY complete_supported DESC, (source_mode = 'transcript_import') DESC,
                 covered_through DESC NULLS LAST, recorded_at DESC, rowid DESC
    ) AS priority
    FROM capture_observations
), metadata AS (
    SELECT session_id,
        arg_max(session_metadata.model::VARCHAR, (recorded_at, rowid)) AS model,
        arg_max(session_metadata.project::VARCHAR, (recorded_at, rowid)) AS project,
        arg_max(session_metadata, (recorded_at, rowid)) AS effective_metadata
    FROM capture_observations GROUP BY session_id
)
SELECT s.session_id, s.agent_id, a.capture_tier,
    CASE
        WHEN o.complete_supported AND o.covered_through >= c.last_event_at THEN 'full'
        WHEN o.source_mode = 'transcript_import' THEN 'transcript_import'
        WHEN o.observation_id IS NOT NULL THEN 'partial'
        WHEN s.capture_completeness IN ('full', 'transcript_import')
            THEN 'unverified_legacy_assertion'
        ELSE 'partial'
    END AS capture_completeness,
    s.capture_completeness AS recorded_capture_completeness,
    s.harness_session_id,
    coalesce(md.model, s.model) AS model,
    s.scope, coalesce(md.project, s.project) AS project, s.started_at,
    c.ended_at, c.last_event_at,
    coalesce(c.event_count, 0) AS event_count, coalesce(c.answers, 0) AS answers,
    coalesce(c.recalls, 0) AS recalls, coalesce(c.compactions, 0) AS compactions,
    coalesce(c.capture_errors, 0) AS capture_errors,
    o.observation_id AS coverage_observation_id,
    o.source_mode AS coverage_source_mode,
    o.covered_from AS coverage_from, o.covered_through AS coverage_through,
    o.cursor_start AS coverage_cursor_start, o.cursor_end AS coverage_cursor_end,
    o.range_start AS coverage_range_start, o.range_end AS coverage_range_end,
    o.exclusions AS coverage_exclusions, o.parse_errors AS coverage_parse_errors,
    (o.covered_through IS NULL OR c.last_event_at > o.covered_through
        OR coalesce(json_array_length(o.parse_errors::JSON), 0) > 0) AS coverage_gap,
    md.effective_metadata
FROM sessions s
JOIN agents a USING (agent_id)
LEFT JOIN counts c USING (session_id)
LEFT JOIN observed o ON o.session_id = s.session_id AND o.priority = 1
LEFT JOIN metadata md ON md.session_id = s.session_id;

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
            ' ?insert into (main\.)?(sessions|events|memories|memory_sources|memory_actions|retrievals|retrieval_items|context_links|memory_deduplications|memory_replacements|memory_correction_work|memory_correction_closures|capture_observations)\b.*'))
);
