PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_key TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL CHECK (schema_version = 'vv-agent.checkpoint.v3'),
    run_definition_schema TEXT NOT NULL CHECK (run_definition_schema = 'vv-agent.run-definition.v2'),
    run_definition TEXT NOT NULL,
    task_id TEXT NOT NULL,
    root_run_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    run_definition_digest TEXT NOT NULL,
    resume_attempt INTEGER NOT NULL CHECK (resume_attempt >= 1),
    cycle_index INTEGER NOT NULL CHECK (cycle_index >= 0),
    status TEXT NOT NULL,
    messages TEXT NOT NULL,
    cycles TEXT NOT NULL,
    model_calls TEXT NOT NULL,
    shared_state TEXT NOT NULL,
    budget_usage TEXT,
    event_cursor TEXT,
    event_outbox TEXT NOT NULL,
    extension_state TEXT NOT NULL,
    model_call_journal TEXT NOT NULL,
    tool_journal TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    claim_token TEXT,
    claimed_cycle INTEGER,
    lease_expires_at_ms INTEGER,
    terminal_result TEXT,
    terminal_acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (terminal_acknowledged IN (0, 1)),
    CHECK (
        (claim_token IS NULL AND claimed_cycle IS NULL AND lease_expires_at_ms IS NULL)
        OR
        (claim_token IS NOT NULL AND claimed_cycle IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
    ),
    CHECK (claim_token IS NULL OR claimed_cycle = cycle_index + 1),
    CHECK (terminal_result IS NULL OR claim_token IS NULL)
);

CREATE INDEX IF NOT EXISTS checkpoints_status_idx
    ON checkpoints(status);
