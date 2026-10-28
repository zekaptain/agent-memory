-- 0001_init: append-only ledger, curated memories, retrieval provenance.
--
-- Every table is append-only. UPDATE and DELETE are rejected by triggers, so
-- the history cannot be rewritten by any client. State that changes over time
-- (memory status) is derived from append-only action rows through views.
--
-- DuckDB 2.0 constraints this schema is shaped around:
--   * A trigger cannot modify its own table ("recursive trigger chains").
--   * Trigger bodies must be DML, so guards raise via INSERT ... SELECT error().
--   * INSERT OR IGNORE / ON CONFLICT run as MERGE, which is rejected on tables
--     with triggers. Idempotent writes use INSERT ... WHERE NOT EXISTS.

CREATE TABLE _append_only_guard (msg VARCHAR);

-- Vocabulary tables. New values are added by migration, not by clients.

CREATE TABLE agents (
    agent_id      VARCHAR PRIMARY KEY,
    display_name  VARCHAR NOT NULL,
    capture_tier  VARCHAR NOT NULL CHECK (capture_tier IN ('A', 'B', 'C')),
    integration   VARCHAR NOT NULL,
    notes         VARCHAR,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO agents (agent_id, display_name, capture_tier, integration, notes) VALUES
    ('factory-droid',   'Factory Droid',   'A', 'hooks+mcp',     'hooks.json: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, PreCompact, Stop, SubagentStop, Notification, SessionEnd. Hooks get transcript_path; no hook carries the assistant reply, so replies come from the transcript.'),
    ('mistral-vibe',    'Mistral Vibe',    'A', 'hooks+mcp',     'hooks.toml: pre_tool, post_tool, post_agent only; no prompt, session, or compaction hooks. Hooks get transcript_path, so prompts and replies come from the transcript. Project hooks need a trusted folder; user hooks always load.'),
    ('oh-my-pi',        'oh-my-pi (omp)',  'A', 'extension+mcp', 'In-process TypeScript hooks: session, agent, turn, message, tool, approval, and compaction events, with message text. Complete capture without transcript import.'),
    ('amp',             'Amp',             'B', 'plugin+mcp',    'In-process plugin API: session.start, agent.start (prompt), tool.call, tool.result, agent.end (every message of the turn). No compaction or session-end events. Threads also stored server-side.'),
    ('jetbrains-junie', 'JetBrains Junie', 'C', 'mcp',           'No hooks. Capture limited to what the agent logs through MCP tools.'),
    ('human',           'Human (mem CLI)', 'A', 'cli',           'Direct use of the mem CLI.');

CREATE TABLE event_kinds (
    kind         VARCHAR PRIMARY KEY,
    description  VARCHAR NOT NULL
);

INSERT INTO event_kinds VALUES
    ('session_start',  'Harness session began'),
    ('session_end',    'Harness session ended'),
    ('user_msg',       'Message from the user'),
    ('assistant_msg',  'Answer or message from the agent'),
    ('system_context', 'Instructions or context injected by the harness or hooks'),
    ('tool_call',      'Agent invoked a tool'),
    ('tool_result',    'Result returned by a tool'),
    ('memory_recall',  'Agent requested memories'),
    ('memory_write',   'Agent wrote or changed a memory'),
    ('compaction',     'Harness summarized the context window'),
    ('decision',       'Explicit decision recorded by the agent or user'),
    ('outcome',        'Result of the work (tests passed, PR merged, ...)'),
    ('capture_error',  'Capture pipeline failed; raw record preserved in spool'),
    ('note',           'Free-form annotation');

CREATE TABLE memory_types (
    type         VARCHAR PRIMARY KEY,
    description  VARCHAR NOT NULL
);

INSERT INTO memory_types VALUES
    ('fact',        'Statement about the world, a codebase, or a system'),
    ('preference',  'How the user wants things done'),
    ('decision',    'A choice that was made, with its rationale'),
    ('procedure',   'How to do something; steps or commands'),
    ('observation', 'Something noticed during work, lower confidence'),
    ('profile',     'Stable attribute of the user');

CREATE TABLE memory_action_types (
    action       VARCHAR PRIMARY KEY,
    description  VARCHAR NOT NULL
);

INSERT INTO memory_action_types VALUES
    ('quarantine',         'Hide from recall until reviewed'),
    ('approve',            'Reviewed and allowed into recall'),
    ('retire',             'No longer true or useful; permanently out of recall'),
    ('confirm',            'Re-observed or re-stated; evidence it still holds'),
    ('flag_contradiction', 'Conflicts with related_memory_id');

-- Ledger

CREATE TABLE sessions (
    session_id            UUID PRIMARY KEY,
    agent_id              VARCHAR NOT NULL REFERENCES agents (agent_id),
    harness_session_id    VARCHAR,
    harness_version       VARCHAR,
    model                 VARCHAR,
    scope                 VARCHAR NOT NULL DEFAULT 'global',
    project               VARCHAR,
    cwd                   VARCHAR,
    capture_completeness  VARCHAR NOT NULL
        CHECK (capture_completeness IN ('full', 'partial', 'transcript_import')),
    started_at            TIMESTAMPTZ NOT NULL,
    recorded_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata              VARIANT,
    -- Harnesses fire "session start" more than once per session (resume,
    -- clear, after compaction). One ledger session per harness session.
    UNIQUE (agent_id, harness_session_id)
);

-- seq is the capturer's best-effort order within a session. Hooks can run
-- concurrently, so it may collide or skip; order audits by (ts, recorded_at).
CREATE TABLE events (
    event_id         UUID PRIMARY KEY,
    session_id       UUID NOT NULL REFERENCES sessions (session_id),
    seq              INTEGER NOT NULL,
    kind             VARCHAR NOT NULL REFERENCES event_kinds (kind),
    role             VARCHAR
        CHECK (role IS NULL OR role IN ('user', 'assistant', 'system', 'tool', 'memory', 'harness')),
    content          VARCHAR,
    content_hash     VARCHAR,
    content_bytes    BIGINT,
    truncated        BOOLEAN NOT NULL DEFAULT false,
    blob_ref         VARCHAR,
    payload          VARIANT,
    parent_event_id  UUID REFERENCES events (event_id),
    token_estimate   INTEGER,
    redactions       INTEGER NOT NULL DEFAULT 0,
    ts               TIMESTAMPTZ NOT NULL,
    recorded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Memory

CREATE TABLE memories (
    memory_id           UUID PRIMARY KEY,
    scope               VARCHAR NOT NULL,
    type                VARCHAR NOT NULL REFERENCES memory_types (type),
    key                 VARCHAR,
    content             VARCHAR NOT NULL,
    content_hash        VARCHAR NOT NULL,
    confidence          DOUBLE NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    importance          DOUBLE NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
    trust_level         VARCHAR NOT NULL CHECK (trust_level IN ('user', 'agent', 'tool', 'imported')),
    supersedes          UUID REFERENCES memories (memory_id),
    valid_from          TIMESTAMPTZ,
    valid_to            TIMESTAMPTZ,
    expires_at          TIMESTAMPTZ,
    created_by_session  UUID REFERENCES sessions (session_id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata            VARIANT
);

CREATE TABLE memory_sources (
    memory_id   UUID NOT NULL REFERENCES memories (memory_id),
    event_id    UUID NOT NULL REFERENCES events (event_id),
    relation    VARCHAR NOT NULL
        CHECK (relation IN ('derived_from', 'user_stated', 'supports', 'contradicts')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, event_id, relation)
);

CREATE TABLE memory_actions (
    action_id          UUID PRIMARY KEY,
    memory_id          UUID NOT NULL REFERENCES memories (memory_id),
    action             VARCHAR NOT NULL REFERENCES memory_action_types (action),
    actor              VARCHAR NOT NULL,
    session_id         UUID REFERENCES sessions (session_id),
    related_memory_id  UUID REFERENCES memories (memory_id),
    reason             VARCHAR,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Retrieval provenance

CREATE TABLE retrievals (
    retrieval_id      UUID PRIMARY KEY,
    session_id        UUID NOT NULL REFERENCES sessions (session_id),
    request_event_id  UUID REFERENCES events (event_id),
    query_text        VARCHAR,
    scopes            VARCHAR[],
    strategy          VARCHAR NOT NULL,
    sql_used          VARCHAR,
    params            VARIANT,
    candidates        INTEGER,
    returned          INTEGER NOT NULL,
    token_estimate    INTEGER,
    latency_ms        DOUBLE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE retrieval_items (
    retrieval_id      UUID NOT NULL REFERENCES retrievals (retrieval_id),
    rank              INTEGER NOT NULL,
    memory_id         UUID NOT NULL REFERENCES memories (memory_id),
    score             DOUBLE,
    score_components  VARIANT,
    injected          BOOLEAN NOT NULL,
    status_at_recall  VARCHAR
        CHECK (status_at_recall IS NULL OR status_at_recall IN
               ('active', 'quarantined', 'superseded', 'expired', 'retired')),
    PRIMARY KEY (retrieval_id, rank)
);

-- link_id lets a replayed capture skip links it already wrote. Capturers
-- should derive it from the answer and target ids; the default is for ad hoc use.
CREATE TABLE context_links (
    link_id           UUID PRIMARY KEY DEFAULT uuidv7(),
    answer_event_id   UUID NOT NULL REFERENCES events (event_id),
    link_type         VARCHAR NOT NULL CHECK (link_type IN ('retrieval', 'cited_memory', 'context_event')),
    retrieval_id      UUID REFERENCES retrievals (retrieval_id),
    memory_id         UUID REFERENCES memories (memory_id),
    context_event_id  UUID REFERENCES events (event_id),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (
        (link_type = 'retrieval'     AND retrieval_id     IS NOT NULL) OR
        (link_type = 'cited_memory'  AND memory_id        IS NOT NULL) OR
        (link_type = 'context_event' AND context_event_id IS NOT NULL)
    )
);

-- Capture pre-reads a session's event ids before every write; remember() dedupes
-- on (scope, content_hash).
CREATE INDEX events_session_ts ON events (session_id, ts);
CREATE INDEX memories_scope_hash ON memories (scope, content_hash);

-- Append-only guards. Statement-level AFTER triggers with an OLD TABLE, so a
-- zero-row UPDATE/DELETE is harmless and any real one aborts the statement.

CREATE TRIGGER schema_migrations_no_update AFTER UPDATE ON schema_migrations REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('schema_migrations is append-only') FROM o;
CREATE TRIGGER schema_migrations_no_delete AFTER DELETE ON schema_migrations REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('schema_migrations is append-only') FROM o;
CREATE TRIGGER agents_no_update AFTER UPDATE ON agents REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('agents is append-only') FROM o;
CREATE TRIGGER agents_no_delete AFTER DELETE ON agents REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('agents is append-only') FROM o;
CREATE TRIGGER event_kinds_no_update AFTER UPDATE ON event_kinds REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('event_kinds is append-only') FROM o;
CREATE TRIGGER event_kinds_no_delete AFTER DELETE ON event_kinds REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('event_kinds is append-only') FROM o;
CREATE TRIGGER memory_types_no_update AFTER UPDATE ON memory_types REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_types is append-only') FROM o;
CREATE TRIGGER memory_types_no_delete AFTER DELETE ON memory_types REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_types is append-only') FROM o;
CREATE TRIGGER memory_action_types_no_update AFTER UPDATE ON memory_action_types REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_action_types is append-only') FROM o;
CREATE TRIGGER memory_action_types_no_delete AFTER DELETE ON memory_action_types REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_action_types is append-only') FROM o;
CREATE TRIGGER sessions_no_update AFTER UPDATE ON sessions REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('sessions is append-only') FROM o;
CREATE TRIGGER sessions_no_delete AFTER DELETE ON sessions REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('sessions is append-only') FROM o;
CREATE TRIGGER events_no_update AFTER UPDATE ON events REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('events is append-only') FROM o;
CREATE TRIGGER events_no_delete AFTER DELETE ON events REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('events is append-only') FROM o;
CREATE TRIGGER memories_no_update AFTER UPDATE ON memories REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memories is append-only') FROM o;
CREATE TRIGGER memories_no_delete AFTER DELETE ON memories REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memories is append-only') FROM o;
CREATE TRIGGER memory_sources_no_update AFTER UPDATE ON memory_sources REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_sources is append-only') FROM o;
CREATE TRIGGER memory_sources_no_delete AFTER DELETE ON memory_sources REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_sources is append-only') FROM o;
CREATE TRIGGER memory_actions_no_update AFTER UPDATE ON memory_actions REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_actions is append-only') FROM o;
CREATE TRIGGER memory_actions_no_delete AFTER DELETE ON memory_actions REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_actions is append-only') FROM o;
CREATE TRIGGER retrievals_no_update AFTER UPDATE ON retrievals REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('retrievals is append-only') FROM o;
CREATE TRIGGER retrievals_no_delete AFTER DELETE ON retrievals REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('retrievals is append-only') FROM o;
CREATE TRIGGER retrieval_items_no_update AFTER UPDATE ON retrieval_items REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('retrieval_items is append-only') FROM o;
CREATE TRIGGER retrieval_items_no_delete AFTER DELETE ON retrieval_items REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('retrieval_items is append-only') FROM o;
CREATE TRIGGER context_links_no_update AFTER UPDATE ON context_links REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('context_links is append-only') FROM o;
CREATE TRIGGER context_links_no_delete AFTER DELETE ON context_links REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('context_links is append-only') FROM o;

-- Full-text search over memories. incremental = 1 maintains the index with
-- triggers, which coexist with the guards above (same statement granularity).
PRAGMA create_fts_index('memories', 'memory_id', 'content', 'key', incremental = 1);

-- Derived state

-- Precedence: retired > superseded > expired > latest quarantine/approve >
-- default by trust (memories derived from tool output start quarantined,
-- because tool output is the main prompt-injection channel).
--
-- created_at defaults to now(), which is fixed per transaction, so actions
-- written together tie. rowid breaks the tie: tables are append-only, so it
-- follows insertion order within a transaction and commit order across them.
CREATE VIEW memory_status AS
WITH review AS (
    SELECT memory_id, arg_max(action, (created_at, rowid)) AS last_review
    FROM memory_actions
    WHERE action IN ('quarantine', 'approve')
    GROUP BY memory_id
),
retired AS (
    SELECT DISTINCT memory_id FROM memory_actions WHERE action = 'retire'
),
superseded AS (
    SELECT supersedes AS memory_id, min(created_at) AS superseded_at
    FROM memories
    WHERE supersedes IS NOT NULL
    GROUP BY supersedes
)
SELECT
    m.memory_id,
    CASE
        WHEN r.memory_id IS NOT NULL THEN 'retired'
        WHEN s.memory_id IS NOT NULL THEN 'superseded'
        WHEN m.expires_at IS NOT NULL AND m.expires_at <= now() THEN 'expired'
        WHEN v.last_review = 'quarantine' THEN 'quarantined'
        WHEN v.last_review = 'approve' THEN 'active'
        WHEN m.trust_level = 'tool' THEN 'quarantined'
        ELSE 'active'
    END AS status,
    v.last_review,
    s.superseded_at
FROM memories m
LEFT JOIN review v USING (memory_id)
LEFT JOIN retired r USING (memory_id)
LEFT JOIN superseded s USING (memory_id);

CREATE VIEW active_memories AS
SELECT m.*
FROM memories m
JOIN memory_status st USING (memory_id)
WHERE st.status = 'active';

-- One chronological feed of every memory change.
CREATE VIEW memory_history AS
SELECT
    m.created_at AS ts,
    m.memory_id,
    CASE WHEN m.supersedes IS NULL THEN 'created' ELSE 'created_superseding' END AS event,
    coalesce(s.agent_id, 'unknown') AS actor,
    m.created_by_session AS session_id,
    m.supersedes AS related_memory_id,
    NULL::VARCHAR AS reason
FROM memories m
LEFT JOIN sessions s ON s.session_id = m.created_by_session
UNION ALL
SELECT a.created_at, a.memory_id, a.action, a.actor, a.session_id, a.related_memory_id, a.reason
FROM memory_actions a;

CREATE VIEW session_summary AS
SELECT
    s.session_id,
    s.agent_id,
    a.capture_tier,
    s.capture_completeness,
    s.harness_session_id,
    s.model,
    s.scope,
    s.project,
    s.started_at,
    max(e.ts) FILTER (WHERE e.kind = 'session_end') AS ended_at,
    max(e.ts) AS last_event_at,
    count(e.event_id) AS event_count,
    count(e.event_id) FILTER (WHERE e.kind = 'assistant_msg') AS answers,
    count(e.event_id) FILTER (WHERE e.kind = 'memory_recall') AS recalls,
    count(e.event_id) FILTER (WHERE e.kind = 'compaction') AS compactions,
    count(e.event_id) FILTER (WHERE e.kind = 'capture_error') AS capture_errors
FROM sessions s
JOIN agents a USING (agent_id)
LEFT JOIN events e USING (session_id)
GROUP BY ALL;

-- "Why did the agent say that?": one row per (answer, item it was shown,
-- evidence behind that item). Filter by answer_event_id.
CREATE VIEW why_answer AS
WITH shown AS (
    SELECT cl.answer_event_id, 'retrieval' AS via, r.retrieval_id, r.query_text, r.strategy,
           ri.rank, ri.score, ri.injected, ri.status_at_recall, ri.memory_id,
           NULL::UUID AS context_event_id
    FROM context_links cl
    JOIN retrievals r ON r.retrieval_id = cl.retrieval_id
    JOIN retrieval_items ri ON ri.retrieval_id = r.retrieval_id
    WHERE cl.link_type = 'retrieval'
    UNION ALL
    SELECT cl.answer_event_id, 'cited_memory', NULL, NULL, NULL, NULL, NULL, NULL, NULL, cl.memory_id, NULL
    FROM context_links cl
    WHERE cl.link_type = 'cited_memory'
    UNION ALL
    SELECT cl.answer_event_id, 'context_event', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, cl.context_event_id
    FROM context_links cl
    WHERE cl.link_type = 'context_event'
)
SELECT
    sh.answer_event_id,
    a.session_id AS answer_session_id,
    a.ts AS answered_at,
    sh.via,
    sh.retrieval_id,
    sh.query_text,
    sh.strategy,
    sh.rank,
    sh.score,
    sh.injected,
    EXISTS (
        SELECT 1 FROM context_links c2
        WHERE c2.answer_event_id = sh.answer_event_id
          AND c2.link_type = 'cited_memory'
          AND c2.memory_id = sh.memory_id
    ) AS cited,
    m.memory_id,
    m.type AS memory_type,
    m.scope AS memory_scope,
    m.content AS memory_content,
    m.trust_level,
    sh.status_at_recall,
    st.status AS memory_status_now,
    ms.relation AS source_relation,
    src.event_id AS source_event_id,
    src.kind AS source_kind,
    src.ts AS source_ts,
    src.content AS source_content,
    src.session_id AS source_session_id,
    src_s.agent_id AS source_agent_id
FROM shown sh
JOIN events a ON a.event_id = sh.answer_event_id
LEFT JOIN memories m ON m.memory_id = sh.memory_id
LEFT JOIN memory_status st ON st.memory_id = m.memory_id
LEFT JOIN memory_sources ms ON ms.memory_id = m.memory_id
LEFT JOIN events src ON src.event_id = coalesce(ms.event_id, sh.context_event_id)
LEFT JOIN sessions src_s ON src_s.session_id = src.session_id;

-- Quack authorization policy, evaluated by the server for every client
-- statement. Defense in depth: guards and enable_external_access = false also
-- apply. Clients send no bound parameters (Quack 2.0.0 does not forward them);
-- values travel as streamed rows, so legitimate SQL text has no user content.
--
-- Normalization removes quoted literals and identifiers so keywords inside
-- data cannot trigger or evade the checks. Constructs the stripper does not
-- model (comments, dollar quotes, E'' strings) are rejected outright, so the
-- stripped text always matches how DuckDB tokenizes the statement.
CREATE MACRO mem_sql_normalize(q) AS
    regexp_replace(
        lower(regexp_replace(q, '''(?:[^'']|'''')*''|"(?:[^"]|"")*"', '''''', 'g')),
        '\s+', ' ', 'g');

CREATE MACRO mem_authorize(sid, q) AS (
    regexp_full_match(
        mem_sql_normalize(q),
        ' ?(select|with|from|insert|begin|commit|rollback|abort|end|describe|show|explain|summarize)\b.*')
    AND NOT regexp_matches(mem_sql_normalize(q), '--|/\*|\$[a-z0-9_]*\$|\be''''')
    AND NOT regexp_matches(regexp_replace(mem_sql_normalize(q), '[; ]+$', ''), ';')
    AND NOT regexp_matches(
        mem_sql_normalize(q),
        '\b(update|delete|drop|alter|create|attach|detach|copy|export|import|install|load|pragma|set|reset|call|vacuum|checkpoint|force_checkpoint|truncate|merge|grant|revoke|use|connect|disconnect|prepare|execute|deallocate|query|query_table|enable_logging|disable_logging|truncate_duckdb_logs|enable_profiling|disable_profiling)\b')
    AND NOT regexp_matches(mem_sql_normalize(q), '\bquack_[a-z0-9_]*\s*\(')
    AND (
        NOT regexp_matches(mem_sql_normalize(q), '\binsert\b')
        OR regexp_full_match(
            mem_sql_normalize(q),
            ' ?insert into (main\.)?(sessions|events|memories|memory_sources|memory_actions|retrievals|retrieval_items|context_links)\b.*')
    )
);
