"""Owner-bound logical-state reads for exploration evidence."""

from vv_agent.kernel._boundary_hydration import hydrate_session_records


def read_logical_state(world, session_id="s"):
    """Read witness state through the store's tenant/session content boundary."""

    state, rows, inbox_seq = world.read_state(session_id)
    return state, hydrate_session_records(rows, world.hydration(session_id)), inbox_seq


def retained_definition(world, session_id="s"):
    """Return the hydrated frozen definition declared by turn start."""

    _, rows, _ = read_logical_state(world, session_id)
    started = next(row.record for row in rows if row.record.kind == "turn_started")
    return started.payload["definition"]
