-- Admission writes lessons into production memory. It gets its own agent, so that the
-- judge agent stays what migration 0005 registered it for: experiment verdicts only.
INSERT INTO agents (agent_id, display_name, capture_tier, integration, notes) VALUES
    ('admission', 'Admission (mem admit)', 'A', 'cli', 'Admission decisions and the lessons they admit.');
