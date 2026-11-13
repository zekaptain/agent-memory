-- A session's row holds what the first record the ledger received of the session knew. For
-- a session captured through hooks that record is a hook's, which gives no title, no model
-- and no harness version. The transcript import that follows says all three in its
-- observation, and hooks go on writing observations after it. The summary took a session's
-- metadata from its latest observation, so a later hook replaced what an import had said.
--
-- The summary now takes each fact about a session from the observations of its transcript
-- imports, the latest that states the fact; failing that from the latest other observation
-- that states it; failing that from the row. A session's title, the session that called
-- it, and its harness version get columns of their own. A caller that neither an
-- observation nor the row names is the one the session's start events name: a start hook's
-- payload has always held it, and its observation has not.
CREATE OR REPLACE VIEW session_summary AS
WITH counts AS (
    SELECT session_id,
        min(ts) AS first_event_at,
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
), said AS (
    SELECT session_id, session_metadata AS meta,
        (source_mode = 'transcript_import', recorded_at, rowid) AS standing
    FROM capture_observations
), metadata AS (
    SELECT session_id,
        arg_max(meta.model::VARCHAR, standing) AS model,
        arg_max(meta.project::VARCHAR, standing) AS project,
        arg_max(meta.harness_version::VARCHAR, standing) AS harness_version,
        arg_max(meta.metadata.title::VARCHAR, standing) AS title,
        arg_max(meta.metadata.calling_session_id::VARCHAR, standing) AS calling_session_id,
        arg_max(meta, standing) AS effective_metadata
    FROM said GROUP BY session_id
), started AS (
    SELECT session_id, max(payload.calling_session_id::VARCHAR) AS calling_session_id
    FROM events WHERE kind = 'session_start' GROUP BY session_id
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
    s.scope, coalesce(md.project, s.project) AS project,
    least(s.started_at, coalesce(c.first_event_at, s.started_at)) AS started_at,
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
    md.effective_metadata,
    s.started_at AS recorded_started_at,
    coalesce(md.title, s.metadata.title::VARCHAR) AS title,
    coalesce(md.calling_session_id, s.metadata.calling_session_id::VARCHAR,
        st.calling_session_id) AS calling_session_id,
    coalesce(md.harness_version, s.harness_version) AS harness_version
FROM sessions s
JOIN agents a USING (agent_id)
LEFT JOIN counts c USING (session_id)
LEFT JOIN observed o ON o.session_id = s.session_id AND o.priority = 1
LEFT JOIN metadata md ON md.session_id = s.session_id
LEFT JOIN started st ON st.session_id = s.session_id;
