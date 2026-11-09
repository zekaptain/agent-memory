-- Original placeholders/context links remain immutable, including zero-result searches.
CREATE TABLE answer_bindings (
    binding_id UUID PRIMARY KEY,
    placeholder_event_id UUID NOT NULL REFERENCES events (event_id),
    answer_event_id UUID REFERENCES events (event_id),
    supersedes_binding_id UUID UNIQUE REFERENCES answer_bindings (binding_id),
    root_placeholder_id UUID UNIQUE,
    evidence VARIANT NOT NULL,
    resolution_method VARCHAR NOT NULL,
    reason_code VARCHAR NOT NULL,
    captured_retrieval_ids UUID[] NOT NULL DEFAULT [],
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (placeholder_event_id <> answer_event_id),
    CHECK ((supersedes_binding_id IS NULL AND root_placeholder_id IS NOT NULL
            AND root_placeholder_id = placeholder_event_id)
        OR (supersedes_binding_id IS NOT NULL AND root_placeholder_id IS NULL))
);
CREATE INDEX answer_bindings_placeholder ON answer_bindings (placeholder_event_id);
CREATE TRIGGER answer_bindings_no_update AFTER UPDATE ON answer_bindings REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('answer_bindings is append-only') FROM o;
CREATE TRIGGER answer_bindings_no_delete AFTER DELETE ON answer_bindings REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('answer_bindings is append-only') FROM o;
CREATE TRIGGER answer_bindings_validate AFTER INSERT ON answer_bindings REFERENCING NEW TABLE AS n FOR EACH STATEMENT
    INSERT INTO _append_only_guard
    SELECT error('invalid answer binding: require a placeholder, same-session actual reply, and linear correction')
    FROM n
    JOIN events p ON p.event_id = n.placeholder_event_id
    LEFT JOIN events a ON a.event_id = n.answer_event_id
    LEFT JOIN answer_bindings old ON old.binding_id = n.supersedes_binding_id
    WHERE p.kind <> 'assistant_msg'
       OR NOT coalesce(try_cast(p.payload.placeholder AS BOOLEAN), false)
       OR (n.answer_event_id IS NOT NULL AND
           (a.kind <> 'assistant_msg' OR a.session_id <> p.session_id
            OR coalesce(try_cast(a.payload.placeholder AS BOOLEAN), false)))
       OR (n.supersedes_binding_id IS NOT NULL
           AND old.placeholder_event_id <> n.placeholder_event_id);

CREATE VIEW effective_answer_bindings AS
SELECT b.* FROM answer_bindings b
WHERE NOT EXISTS (SELECT 1 FROM answer_bindings next WHERE next.supersedes_binding_id = b.binding_id);

CREATE VIEW effective_context_links AS
SELECT cl.link_id, coalesce(b.answer_event_id, cl.answer_event_id) AS answer_event_id,
    cl.answer_event_id AS original_answer_event_id, cl.link_type,
    cl.retrieval_id, cl.memory_id, cl.context_event_id, cl.created_at,
    b.binding_id, b.reason_code AS binding_reason,
    CASE WHEN coalesce(try_cast(e.payload.placeholder AS BOOLEAN), false)
         THEN cl.answer_event_id END AS placeholder_event_id
FROM context_links cl
JOIN events e ON e.event_id = cl.answer_event_id
LEFT JOIN effective_answer_bindings b ON b.placeholder_event_id = cl.answer_event_id;

CREATE OR REPLACE VIEW why_answer AS
WITH shown AS (
    SELECT cl.*, 'retrieval' AS via, r.query_text, r.strategy, r.returned,
           ri.rank, ri.score, ri.injected, ri.status_at_recall, ri.memory_id AS shown_memory_id
    FROM effective_context_links cl
    JOIN retrievals r ON r.retrieval_id = cl.retrieval_id
    LEFT JOIN retrieval_items ri ON ri.retrieval_id = r.retrieval_id
    WHERE cl.link_type = 'retrieval'
    UNION ALL
    SELECT cl.*, 'cited_memory', NULL, NULL, NULL, NULL, NULL, NULL, NULL, cl.memory_id
    FROM effective_context_links cl WHERE cl.link_type = 'cited_memory'
    UNION ALL
    SELECT cl.*, 'context_event', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
    FROM effective_context_links cl WHERE cl.link_type = 'context_event'
)
SELECT sh.answer_event_id, a.session_id AS answer_session_id, a.ts AS answered_at,
    sh.via, sh.retrieval_id, sh.query_text, sh.strategy, sh.rank, sh.score, sh.injected,
    EXISTS (
        SELECT 1 FROM effective_context_links c2
        WHERE c2.answer_event_id = sh.answer_event_id AND c2.link_type = 'cited_memory'
          AND c2.memory_id = sh.shown_memory_id
    ) AS cited,
    m.memory_id, m.type AS memory_type, m.scope AS memory_scope,
    m.content AS memory_content, m.trust_level, sh.status_at_recall,
    st.status AS memory_status_now, ms.relation AS source_relation,
    src.event_id AS source_event_id, src.kind AS source_kind, src.ts AS source_ts,
    src.content AS source_content, src.session_id AS source_session_id,
    src_s.agent_id AS source_agent_id,
    sh.link_id AS original_link_id, sh.original_answer_event_id, sh.placeholder_event_id,
    sh.binding_id, sh.binding_reason, sh.returned,
    CASE WHEN sh.returned = 0 THEN 'miss'
         WHEN sh.returned > 0 THEN 'returned' END AS recall_outcome,
    CASE WHEN sh.binding_id IS NOT NULL AND sh.retrieval_id IS NOT NULL
         AND EXISTS (
             SELECT 1 FROM answer_bindings b WHERE b.binding_id = sh.binding_id
             AND list_contains(b.captured_retrieval_ids, sh.retrieval_id)
         ) THEN true END AS captured_in_context,
    EXISTS (
        SELECT 1 FROM effective_context_links c2
        WHERE c2.answer_event_id = sh.answer_event_id AND c2.link_type = 'cited_memory'
          AND c2.memory_id = sh.shown_memory_id
    ) AS declared_used
FROM shown sh
JOIN events a ON a.event_id = sh.answer_event_id
LEFT JOIN memories m ON m.memory_id = sh.shown_memory_id
LEFT JOIN memory_status st ON st.memory_id = m.memory_id
LEFT JOIN memory_sources ms ON ms.memory_id = m.memory_id
LEFT JOIN events src ON src.event_id = coalesce(ms.event_id, sh.context_event_id)
LEFT JOIN sessions src_s ON src_s.session_id = src.session_id;

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
            ' ?insert into (main\.)?(sessions|events|memories|memory_sources|memory_actions|retrievals|retrieval_items|context_links|memory_deduplications|memory_replacements|memory_correction_work|memory_correction_closures|capture_observations|answer_bindings)\b.*'))
);
