PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- SQLite enforces scalar lifecycle/fence relations below.  The strict v9 codec
-- (not SQLite JSON1) validates closed HostInteractionRequest and
-- HostInteractionResponse objects, RFC 8785 digests, forbidden fields, and
-- the 65,536-byte UTF-8 limits before any CAS transaction begins.

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_key TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL CHECK (schema_version = 'vv-agent.checkpoint.v9'),
    run_definition_schema TEXT NOT NULL CHECK (run_definition_schema = 'vv-agent.run-definition.v5'),
    run_definition TEXT NOT NULL,
    task_id TEXT NOT NULL,
    root_run_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    run_definition_digest TEXT NOT NULL,
    resume_attempt INTEGER NOT NULL CHECK (resume_attempt >= 1),
    cycle_index INTEGER NOT NULL CHECK (cycle_index >= 0),
    status TEXT NOT NULL,
    cancel_requested INTEGER NOT NULL CHECK (cancel_requested IN (0, 1)),
    active_host_interaction TEXT,
    suspended_origin TEXT,
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
    CHECK (status <> 'deferred' OR (claim_token IS NULL AND claimed_cycle IS NULL AND lease_expires_at_ms IS NULL)),
    CHECK (status <> 'deferred' OR tool_journal <> '[]'),
    CHECK (
        (claim_token IS NULL AND claimed_cycle IS NULL AND lease_expires_at_ms IS NULL)
        OR
        (claim_token IS NOT NULL AND claimed_cycle IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
    ),
    CHECK (claim_token IS NULL OR claimed_cycle = cycle_index + 1),
    CHECK (terminal_result IS NULL OR claim_token IS NULL),
    CHECK (
        (status = 'host_interaction' AND active_host_interaction IS NOT NULL AND suspended_origin IS NULL)
        OR
        (status = 'suspended' AND active_host_interaction IS NULL AND suspended_origin IS NOT NULL)
        OR
        (status NOT IN ('host_interaction', 'suspended') AND active_host_interaction IS NULL AND suspended_origin IS NULL)
    ),
    CHECK (
        status NOT IN ('host_interaction', 'suspended')
        OR (claim_token IS NULL AND claimed_cycle IS NULL AND lease_expires_at_ms IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS checkpoints_status_idx
    ON checkpoints(status);

CREATE TABLE IF NOT EXISTS host_interaction_records (
    record_id TEXT PRIMARY KEY,
    checkpoint_key TEXT NOT NULL,
    interaction_id TEXT NOT NULL,
    logical_cycle INTEGER NOT NULL CHECK (logical_cycle >= 1),
    request TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('active', 'resolved_pending', 'resolved_claimed', 'consumed')),
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    claim_token TEXT,
    lease_expires_at_ms INTEGER,
    response TEXT,
    response_digest TEXT,
    command_id TEXT,
    resolved_revision INTEGER,
    consumed_revision INTEGER,
    last_error TEXT,
    UNIQUE (checkpoint_key, interaction_id),
    CHECK (
        (claim_token IS NULL AND lease_expires_at_ms IS NULL)
        OR (claim_token IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
    ),
    CHECK (
        (state = 'active' AND response IS NULL AND response_digest IS NULL AND command_id IS NULL)
        OR
        (state IN ('resolved_pending', 'resolved_claimed', 'consumed')
         AND response IS NOT NULL AND response_digest IS NOT NULL AND command_id IS NOT NULL)
    ),
    CHECK (state <> 'resolved_claimed' OR (claim_token IS NOT NULL AND lease_expires_at_ms IS NOT NULL)),
    CHECK (state <> 'resolved_pending' OR claim_token IS NULL),
    CHECK (state <> 'consumed' OR consumed_revision IS NOT NULL),
    FOREIGN KEY (checkpoint_key) REFERENCES checkpoints(checkpoint_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS host_interaction_records_checkpoint_idx
    ON host_interaction_records(checkpoint_key, state);

CREATE INDEX IF NOT EXISTS host_interaction_records_recovery_idx
    ON host_interaction_records(state, lease_expires_at_ms);

-- A resolved_pending/resolved_claimed interaction is a hard recovery barrier.
-- SQLite cannot express the cross-table invariant in a CHECK constraint: a
-- resolved_claimed row may be written only by
-- claim_and_consume_host_interaction_response, in the same transaction that
-- locks this row and sets checkpoints.claim_token/claimed_cycle.  A
-- record-only claim is invalid.  The operation injects the response, writes
-- the consumed RunEvent, increments revision relative to admission, marks
-- consumed, releases the transient record claim, and retains the checkpoint
-- execution claim for model/tool ownership.  Rollback before commit leaves
-- the record resolved_pending and the checkpoint at admission_revision.

-- This UI notification outbox is deliberately independent from
-- controller_command_receipts/recovery_dispatch.  The strict codec validates
-- the sanitized HostInteractionNotification payload and its RFC 8785 digest
-- before this row participates in the producer CAS.
-- The payload contains wait_reason=host_interaction.  Delivery is
-- at-least-once with stable notification_id; an observer must deduplicate
-- retries because a callback crash is ambiguous, not exactly-once.  Ambiguous
-- rows are resolved explicitly to delivered, pending (retry), or aborted; the
-- reaper never blind-retries an uncertain callback.
CREATE TABLE IF NOT EXISTS host_interaction_notification_outbox (
    notification_id TEXT PRIMARY KEY,
    checkpoint_key TEXT NOT NULL,
    record_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    outbox_state TEXT NOT NULL CHECK (outbox_state IN ('pending', 'claimed', 'delivered', 'ambiguous', 'aborted')),
    claim_token TEXT,
    lease_expires_at_ms INTEGER,
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    delivered_at_ms INTEGER,
    aborted_at_ms INTEGER,
    abort_reason TEXT,
    last_error TEXT,
    UNIQUE (checkpoint_key, record_id),
    CHECK (
        (outbox_state = 'claimed' AND claim_token IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
        OR
        (outbox_state <> 'claimed' AND claim_token IS NULL AND lease_expires_at_ms IS NULL)
    ),
    CHECK (
        (outbox_state = 'delivered' AND delivered_at_ms IS NOT NULL AND aborted_at_ms IS NULL AND abort_reason IS NULL)
        OR
        (outbox_state = 'aborted' AND aborted_at_ms IS NOT NULL AND delivered_at_ms IS NULL AND abort_reason IS NOT NULL)
        OR
        (outbox_state NOT IN ('delivered', 'aborted') AND delivered_at_ms IS NULL AND aborted_at_ms IS NULL AND abort_reason IS NULL)
    ),
    FOREIGN KEY (checkpoint_key) REFERENCES checkpoints(checkpoint_key) ON DELETE CASCADE,
    FOREIGN KEY (record_id) REFERENCES host_interaction_records(record_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS host_interaction_notification_outbox_checkpoint_idx
    ON host_interaction_notification_outbox(checkpoint_key, outbox_state);

CREATE INDEX IF NOT EXISTS host_interaction_notification_outbox_lease_idx
    ON host_interaction_notification_outbox(outbox_state, lease_expires_at_ms);

CREATE TABLE IF NOT EXISTS deferred_resolution_receipts (
    handle_key TEXT PRIMARY KEY,
    checkpoint_key TEXT NOT NULL,
    handle TEXT NOT NULL,
    result TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_payload_digest TEXT NOT NULL,
    receipt_status TEXT NOT NULL CHECK (receipt_status IN ('succeeded', 'failed')),
    FOREIGN KEY (checkpoint_key) REFERENCES checkpoints(checkpoint_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS deferred_receipts_checkpoint_idx
    ON deferred_resolution_receipts(checkpoint_key);

CREATE TABLE IF NOT EXISTS controller_command_receipts (
    command_id TEXT PRIMARY KEY,
    checkpoint_key TEXT NOT NULL,
    handle TEXT NOT NULL,
    command_digest TEXT NOT NULL,
    command TEXT NOT NULL,
    resume_attempt INTEGER NOT NULL CHECK (resume_attempt >= 1),
    expected_revision INTEGER NOT NULL CHECK (expected_revision >= 0),
    receipt TEXT NOT NULL,
    resulting_status TEXT NOT NULL,
    resulting_revision INTEGER NOT NULL CHECK (resulting_revision >= 0),
    outbox_state TEXT NOT NULL CHECK (outbox_state IN ('pending', 'claimed', 'delivered', 'ambiguous')),
    outbox_id TEXT NOT NULL,
    outbox_action TEXT NOT NULL CHECK (outbox_action IN ('none', 'recovery_dispatch')),
    outbox_destination TEXT,
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    claim_token TEXT,
    lease_expires_at_ms INTEGER,
    delivered_at_ms INTEGER,
    last_error TEXT,
    CHECK (
        (outbox_action = 'none' AND outbox_destination IS NULL)
        OR
        (outbox_action = 'recovery_dispatch' AND outbox_destination = 'distributed_advance')
    ),
    CHECK (outbox_state = 'delivered' OR outbox_action = 'recovery_dispatch'),
    FOREIGN KEY (checkpoint_key) REFERENCES checkpoints(checkpoint_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS controller_command_receipts_checkpoint_idx
    ON controller_command_receipts(checkpoint_key);

CREATE INDEX IF NOT EXISTS controller_command_receipts_outbox_idx
    ON controller_command_receipts(outbox_state, lease_expires_at_ms);
