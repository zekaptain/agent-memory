-- Corrections are proposals, not implicit retirement. Only a durable acceptance
-- hides a predecessor; withdrawal of a successor never reverses that record.
ALTER TABLE memory_actions ADD COLUMN event_id UUID;
ALTER TABLE memory_actions ADD COLUMN metadata VARIANT;

INSERT INTO memory_action_types VALUES
    ('propose_correction', 'Propose memory_id as replacement for related_memory_id'),
    ('merge_lineage', 'Carry source relation aliases into a canonical relation');

CREATE TABLE memory_deduplications (
    memory_id UUID PRIMARY KEY REFERENCES memories(memory_id),
    canonical_id UUID NOT NULL REFERENCES memories(memory_id),
    event_id UUID NOT NULL REFERENCES events(event_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (memory_id <> canonical_id)
);

CREATE TABLE memory_replacements (
    predecessor_id UUID PRIMARY KEY REFERENCES memories(memory_id),
    successor_id UUID NOT NULL REFERENCES memories(memory_id),
    proposal_id UUID NOT NULL REFERENCES memory_actions(action_id),
    evidence_event_id UUID REFERENCES events(event_id),
    evidence_action_id UUID REFERENCES memory_actions(action_id),
    accepted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    method VARCHAR NOT NULL,
    CHECK (predecessor_id <> successor_id),
    CHECK (evidence_event_id IS NOT NULL OR evidence_action_id IS NOT NULL)
);

-- Authorization is recorded once; maintenance cannot manufacture approval.
-- A new approval creates new work, rather than mutating cancelled work.
CREATE TABLE memory_correction_work (
    work_id UUID PRIMARY KEY,
    proposal_id UUID NOT NULL REFERENCES memory_actions(action_id),
    predecessor_id UUID NOT NULL REFERENCES memories(memory_id),
    successor_id UUID NOT NULL REFERENCES memories(memory_id),
    authorization_action_id UUID REFERENCES memory_actions(action_id),
    due_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE memory_correction_closures (
    work_id UUID PRIMARY KEY REFERENCES memory_correction_work(work_id),
    reason VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX memory_work_due ON memory_correction_work(due_at);
CREATE INDEX memory_work_successor ON memory_correction_work(successor_id);
CREATE INDEX memory_actions_target ON memory_actions(memory_id, action);
CREATE INDEX memory_actions_proposal ON memory_actions(related_memory_id, action);
CREATE INDEX memory_replacements_successor ON memory_replacements(successor_id);
CREATE INDEX memory_dedup_canonical ON memory_deduplications(canonical_id);

CREATE VIEW memory_eligibility AS
WITH review AS (
    SELECT memory_id, arg_max(action, (created_at, rowid)) AS last_review,
           arg_max(action_id, (created_at, rowid)) AS review_id
    FROM memory_actions WHERE action IN ('quarantine', 'approve') GROUP BY memory_id
)
SELECT m.memory_id, v.last_review, v.review_id,
       CASE
           WHEN EXISTS (SELECT 1 FROM memory_actions a
                        WHERE a.memory_id = m.memory_id AND a.action = 'retire') THEN 'retired'
           WHEN d.memory_id IS NOT NULL THEN 'deduplicated'
           WHEN r.predecessor_id IS NOT NULL THEN 'superseded'
           WHEN m.expires_at <= now() OR m.valid_to <= now() THEN 'expired'
           WHEN v.last_review = 'quarantine'
                OR (v.last_review IS NULL AND m.trust_level = 'tool') THEN 'quarantined'
           WHEN m.valid_from > now() THEN 'pending'
           ELSE 'active'
       END AS eligibility,
       r.accepted_at AS superseded_at
FROM memories m
LEFT JOIN review v USING (memory_id)
LEFT JOIN memory_replacements r ON r.predecessor_id = m.memory_id
LEFT JOIN memory_deduplications d USING (memory_id);

CREATE OR REPLACE VIEW memory_status AS
SELECT m.memory_id,
       CASE WHEN e.eligibility = 'active' AND m.supersedes IS NOT NULL
                  AND r.predecessor_id IS NULL THEN 'pending'
            ELSE e.eligibility END AS status,
       e.last_review, e.superseded_at,
       CASE WHEN e.eligibility IN ('retired', 'superseded', 'deduplicated', 'expired') THEN NULL
            WHEN m.supersedes IS NOT NULL AND EXISTS (
                 SELECT 1 FROM memory_replacements other
                 WHERE other.predecessor_id = m.supersedes AND other.successor_id <> m.memory_id)
                 THEN 'stale_predecessor'
            WHEN m.supersedes IS NOT NULL AND r.predecessor_id IS NULL
                 AND e.eligibility = 'quarantined' THEN 'awaiting_approval'
            WHEN m.valid_from > now() THEN 'valid_from'
            WHEN m.supersedes IS NOT NULL AND r.predecessor_id IS NULL
                 THEN 'awaiting_acceptance'
            ELSE NULL END AS pending_reason
FROM memories m JOIN memory_eligibility e USING (memory_id)
LEFT JOIN memory_replacements r
  ON r.predecessor_id = m.supersedes AND r.successor_id = m.memory_id;

-- Ordinary proposals and accepted edges both participate in cycle detection.
CREATE VIEW memory_correction_edges AS
SELECT supersedes AS predecessor_id, memory_id AS successor_id
FROM memories WHERE supersedes IS NOT NULL
UNION
SELECT related_memory_id, memory_id FROM memory_actions WHERE action = 'propose_correction';

CREATE VIEW memory_aliases AS
SELECT memory_id, unnest(json_extract_string(metadata::JSON, '$.graph.aliases[*]')) AS alias
FROM memories
UNION
SELECT memory_id, unnest(json_extract_string(metadata::JSON, '$.aliases[*]')) AS alias
FROM memory_actions WHERE action = 'merge_lineage';

CREATE VIEW pending_memory_corrections AS
SELECT w.* FROM memory_correction_work w
WHERE NOT EXISTS (SELECT 1 FROM memory_correction_closures c WHERE c.work_id = w.work_id)
  AND NOT EXISTS (SELECT 1 FROM memory_replacements r WHERE r.predecessor_id = w.predecessor_id);

-- Shared by immediate acceptance and bounded maintenance. The latest review
-- must still be the recorded authorization, not a later confirm or withdrawal.
CREATE VIEW admissible_memory_corrections AS
SELECT w.*, p.event_id AS evidence_event_id,
       CASE WHEN w.authorization_action_id IS NULL THEN 'trust_default'
            ELSE 'approval' END AS method
FROM pending_memory_corrections w
JOIN memory_actions p ON p.action_id = w.proposal_id
JOIN memory_eligibility e ON e.memory_id = w.successor_id
JOIN memories m ON m.memory_id = w.successor_id
WHERE w.due_at <= now() AND e.eligibility = 'active'
  AND (try_cast(json_extract_string(p.metadata::JSON, '$.correction.valid_to') AS TIMESTAMPTZ)
       IS NULL OR now() < cast(json_extract_string(p.metadata::JSON, '$.correction.valid_to') AS TIMESTAMPTZ))
  AND (try_cast(json_extract_string(p.metadata::JSON, '$.correction.expires_at') AS TIMESTAMPTZ)
       IS NULL OR now() < cast(json_extract_string(p.metadata::JSON, '$.correction.expires_at') AS TIMESTAMPTZ))
  AND ((w.authorization_action_id IS NULL AND e.last_review IS NULL
        AND m.trust_level <> 'tool')
       OR (e.last_review = 'approve' AND e.review_id = w.authorization_action_id));

CREATE VIEW memory_proposal_status AS
SELECT p.action_id AS proposal_id, p.related_memory_id AS predecessor_id,
       p.memory_id AS successor_id,
       CASE WHEN r.successor_id = p.memory_id THEN 'accepted'
            WHEN r.successor_id IS NOT NULL THEN 'conflict'
            ELSE 'pending' END AS correction_status,
       CASE WHEN r.successor_id = p.memory_id THEN NULL
            WHEN r.successor_id IS NOT NULL THEN 'stale_predecessor'
            WHEN e.eligibility IN ('retired', 'superseded', 'expired', 'deduplicated')
                 THEN e.eligibility
            WHEN e.eligibility = 'quarantined'
                 OR (json_extract_string(p.metadata::JSON, '$.correction.trust_level') = 'tool'
                     AND coalesce(e.last_review, '') <> 'approve') THEN 'awaiting_approval'
            WHEN greatest(m.valid_from,
                 cast(json_extract_string(p.metadata::JSON, '$.correction.valid_from') AS TIMESTAMPTZ))
                 > now() THEN 'valid_from'
            WHEN cast(json_extract_string(p.metadata::JSON, '$.correction.valid_to') AS TIMESTAMPTZ)
                 <= now() OR cast(json_extract_string(p.metadata::JSON, '$.correction.expires_at') AS TIMESTAMPTZ)
                 <= now() THEN 'expired'
            ELSE 'awaiting_acceptance' END AS pending_reason
FROM memory_actions p
JOIN memories m ON m.memory_id = p.memory_id
JOIN memory_eligibility e ON e.memory_id = p.memory_id
LEFT JOIN memory_replacements r ON r.predecessor_id = p.related_memory_id
WHERE p.action = 'propose_correction';

CREATE OR REPLACE VIEW memory_history AS
SELECT m.created_at AS ts, m.memory_id,
       CASE WHEN m.supersedes IS NULL THEN 'created' ELSE 'created_superseding' END AS event,
       coalesce(s.agent_id, 'unknown') AS actor, m.created_by_session AS session_id,
       m.supersedes AS related_memory_id, NULL::VARCHAR AS reason
FROM memories m LEFT JOIN sessions s ON s.session_id = m.created_by_session
UNION ALL
SELECT a.created_at, a.memory_id, a.action, a.actor, a.session_id, a.related_memory_id, a.reason
FROM memory_actions a
UNION ALL
SELECT r.accepted_at, r.predecessor_id, 'replacement_accepted', a.actor, a.session_id,
       r.successor_id, r.method
FROM memory_replacements r JOIN memory_actions a ON a.action_id = r.proposal_id;

CREATE TRIGGER memories_valid_interval AFTER INSERT ON memories
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO _append_only_guard
SELECT error('invalid memory interval: valid_from must precede valid_to and expires_at')
FROM n WHERE valid_from >= valid_to OR valid_from >= expires_at;

CREATE TRIGGER memories_valid_correction AFTER INSERT ON memories
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO _append_only_guard
SELECT error('invalid correction: self, cross-scope, or stale predecessor; target the accepted successor')
FROM n JOIN memories p ON p.memory_id = n.supersedes
WHERE n.memory_id = n.supersedes OR n.scope <> p.scope
   OR EXISTS (SELECT 1 FROM memory_replacements r WHERE r.predecessor_id = n.supersedes)
   OR EXISTS (SELECT 1 FROM memory_deduplications d WHERE d.memory_id = n.supersedes);

CREATE TRIGGER memories_no_correction_cycle AFTER INSERT ON memories
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO _append_only_guard
WITH RECURSIVE walk(origin, node) AS (
    SELECT memory_id, supersedes FROM n WHERE supersedes IS NOT NULL
    UNION
    SELECT w.origin, e.predecessor_id FROM walk w
    JOIN memory_correction_edges e ON e.successor_id = w.node
)
SELECT error('invalid correction: cycle') FROM walk WHERE origin = node;

CREATE TRIGGER memories_pending_predecessor AFTER INSERT ON memories
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO _append_only_guard
SELECT error('invalid correction: pending predecessor; accept it first or correct its predecessor')
FROM n JOIN memories p ON p.memory_id = n.supersedes
WHERE p.supersedes IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM memory_replacements r
    WHERE r.predecessor_id = p.supersedes AND r.successor_id = p.memory_id);

CREATE TRIGGER actions_valid_proposal AFTER INSERT ON memory_actions
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO _append_only_guard
SELECT error('invalid correction proposal: self, cross-scope, stale predecessor, or missing evidence')
FROM n
LEFT JOIN memories p ON p.memory_id = n.related_memory_id
LEFT JOIN memories s ON s.memory_id = n.memory_id
WHERE n.action = 'propose_correction'
  AND (p.memory_id IS NULL OR n.memory_id = n.related_memory_id OR p.scope <> s.scope
       OR n.event_id IS NULL
       OR NOT EXISTS (SELECT 1 FROM events e WHERE e.event_id = n.event_id)
       OR EXISTS (SELECT 1 FROM memory_deduplications d
                  WHERE d.memory_id IN (n.memory_id, n.related_memory_id))
       OR EXISTS (SELECT 1 FROM memory_replacements r
                  WHERE r.predecessor_id = n.related_memory_id
                    AND r.successor_id <> n.memory_id));

CREATE TRIGGER actions_no_correction_cycle AFTER INSERT ON memory_actions
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO _append_only_guard
WITH RECURSIVE walk(origin, node) AS (
    SELECT memory_id, related_memory_id FROM n WHERE action = 'propose_correction'
    UNION
    SELECT w.origin, e.predecessor_id FROM walk w
    JOIN memory_correction_edges e ON e.successor_id = w.node
)
SELECT error('invalid correction: cycle; target a current successor instead')
FROM walk WHERE origin = node;

CREATE TRIGGER actions_pending_predecessor AFTER INSERT ON memory_actions
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO _append_only_guard
SELECT error('invalid correction: pending predecessor; accept it first or correct its predecessor')
FROM n JOIN memories p ON p.memory_id = n.related_memory_id
WHERE n.action = 'propose_correction' AND p.supersedes IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM memory_replacements r
    WHERE r.predecessor_id = p.supersedes AND r.successor_id = p.memory_id);

CREATE TRIGGER actions_schedule_corrections AFTER INSERT ON memory_actions
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO memory_correction_work
SELECT md5(p.action_id::VARCHAR || ':' || coalesce(e.review_id::VARCHAR, 'trust'))::UUID,
       p.action_id, p.related_memory_id, p.memory_id,
       e.review_id, greatest(coalesce(m.valid_from, now()),
           coalesce(cast(json_extract_string(p.metadata::JSON, '$.correction.valid_from') AS TIMESTAMPTZ), now()),
           now()), now()
FROM memory_actions p
JOIN memories m ON m.memory_id = p.memory_id
JOIN memory_eligibility e ON e.memory_id = p.memory_id
WHERE p.action = 'propose_correction'
  AND (p.action_id IN (SELECT action_id FROM n WHERE action = 'propose_correction')
       OR p.memory_id IN (SELECT memory_id FROM n WHERE action = 'approve'))
  AND e.eligibility IN ('active', 'pending')
  AND (coalesce(json_extract_string(p.metadata::JSON, '$.correction.trust_level'), m.trust_level)
       <> 'tool' OR e.last_review = 'approve')
  AND NOT EXISTS (SELECT 1 FROM memory_replacements r WHERE r.predecessor_id = p.related_memory_id)
  AND NOT EXISTS (
      SELECT 1 FROM memory_correction_work w
      WHERE w.proposal_id = p.action_id
        AND w.authorization_action_id IS NOT DISTINCT FROM e.review_id);

CREATE TRIGGER work_accept_immediately AFTER INSERT ON memory_correction_work
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO memory_replacements
SELECT a.predecessor_id, a.successor_id, a.proposal_id, a.evidence_event_id,
       a.authorization_action_id, now(), a.method
FROM admissible_memory_corrections a JOIN n USING (work_id);

CREATE TRIGGER replacements_validate AFTER INSERT ON memory_replacements
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO _append_only_guard
SELECT error('invalid acceptance: proposal, scope, authorization, validity, or terminal successor')
FROM n
JOIN memories m ON m.memory_id = n.successor_id
JOIN memories p ON p.memory_id = n.predecessor_id
JOIN memory_actions a ON a.action_id = n.proposal_id
WHERE a.action <> 'propose_correction' OR a.memory_id <> n.successor_id
   OR a.related_memory_id <> n.predecessor_id OR m.scope <> p.scope
   OR (
       m.valid_from > now() OR m.valid_to <= now() OR m.expires_at <= now()
       OR cast(json_extract_string(a.metadata::JSON, '$.correction.valid_from') AS TIMESTAMPTZ) > now()
       OR cast(json_extract_string(a.metadata::JSON, '$.correction.valid_to') AS TIMESTAMPTZ) <= now()
       OR cast(json_extract_string(a.metadata::JSON, '$.correction.expires_at') AS TIMESTAMPTZ) <= now()
       OR EXISTS (SELECT 1 FROM memory_eligibility e WHERE e.memory_id = n.successor_id
                  AND e.eligibility <> 'active')
       OR (n.evidence_action_id IS NULL AND
           (m.trust_level = 'tool'
            OR json_extract_string(a.metadata::JSON, '$.correction.trust_level') = 'tool'
            OR EXISTS (
               SELECT 1 FROM memory_eligibility e WHERE e.memory_id = n.successor_id
               AND e.last_review IS NOT NULL)))
       OR (n.evidence_action_id IS NOT NULL AND NOT EXISTS (
           SELECT 1 FROM memory_eligibility e WHERE e.memory_id = n.successor_id
           AND e.last_review = 'approve' AND e.review_id = n.evidence_action_id))
       OR (n.evidence_event_id IS DISTINCT FROM a.event_id));

CREATE TRIGGER deduplications_validate AFTER INSERT ON memory_deduplications
REFERENCING NEW TABLE AS n FOR EACH STATEMENT
INSERT INTO _append_only_guard
SELECT error('invalid deduplication: scope, identity, or canonical link')
FROM n JOIN memories m USING (memory_id)
JOIN memories c ON c.memory_id = n.canonical_id
WHERE m.scope <> c.scope
   OR (m.content_hash <> c.content_hash AND
       (m.key IS NULL OR m.key IS DISTINCT FROM c.key
        OR json_extract(m.metadata::JSON, '$.graph') IS NULL
        OR json_extract(c.metadata::JSON, '$.graph') IS NULL))
   OR EXISTS (SELECT 1 FROM memory_deduplications d WHERE d.memory_id = n.canonical_id)
   OR EXISTS (SELECT 1 FROM memory_replacements r WHERE r.predecessor_id = n.memory_id);

CREATE TRIGGER memory_deduplications_no_update AFTER UPDATE ON memory_deduplications REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_deduplications is append-only') FROM o;
CREATE TRIGGER memory_deduplications_no_delete AFTER DELETE ON memory_deduplications REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_deduplications is append-only') FROM o;
CREATE TRIGGER memory_replacements_no_update AFTER UPDATE ON memory_replacements REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_replacements is append-only') FROM o;
CREATE TRIGGER memory_replacements_no_delete AFTER DELETE ON memory_replacements REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_replacements is append-only') FROM o;
CREATE TRIGGER memory_correction_work_no_update AFTER UPDATE ON memory_correction_work REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_correction_work is append-only') FROM o;
CREATE TRIGGER memory_correction_work_no_delete AFTER DELETE ON memory_correction_work REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_correction_work is append-only') FROM o;
CREATE TRIGGER memory_correction_closures_no_update AFTER UPDATE ON memory_correction_closures REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_correction_closures is append-only') FROM o;
CREATE TRIGGER memory_correction_closures_no_delete AFTER DELETE ON memory_correction_closures REFERENCING OLD TABLE AS o FOR EACH STATEMENT
    INSERT INTO _append_only_guard SELECT error('memory_correction_closures is append-only') FROM o;

CREATE OR REPLACE MACRO mem_authorize(sid, q) AS (
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
            ' ?insert into (main\.)?(sessions|events|memories|memory_sources|memory_actions|retrievals|retrieval_items|context_links|memory_deduplications|memory_replacements|memory_correction_work|memory_correction_closures)\b.*')
    )
);
