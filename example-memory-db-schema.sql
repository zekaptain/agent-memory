CREATE TABLE IF NOT EXISTS episodic_events (
  event_id       UUID,
  user_id        TEXT,
  ts             TIMESTAMP,
  kind           TEXT,          -- "message", "decision", "result"
  source         TEXT,          -- "user", "agent", "tool"
  content        TEXT,
  importance     DOUBLE,        -- 0..1
  expires_at     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_profile_facts (
  user_id        TEXT,
  key            TEXT,          -- "preferred_language"
  value          TEXT,          -- "Hinglish"
  confidence     DOUBLE,        -- 0..1
  updated_at     TIMESTAMP,
  source_event   UUID
);

CREATE TABLE IF NOT EXISTS retrieval_log (
  retrieval_id   UUID,
  user_id        TEXT,
  ts             TIMESTAMP,
  query_type     TEXT,          -- "episodic", "facts", "hybrid"
  sql_used       TEXT,
  items_returned INTEGER,
  token_estimate INTEGER
);

-- query to pull 'high-signal' episodic events within some time frame
SELECT ts, kind, source, content
FROM episodic_events
WHERE user_id = ?
  AND expires_at > NOW()
  AND ts > NOW() - INTERVAL '21 days'
ORDER BY (importance * 0.6) + (1.0 / (1 + DATE_DIFF('hour', ts, NOW())) * 0.4) DESC
LIMIT 12;
