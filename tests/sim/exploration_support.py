"""Owner-bound logical-state reads for exploration evidence."""


def read_logical_state(world, session_id="s"):
    """Read witness state through the store's tenant/session content boundary."""

    return world.read_state(session_id)


def retained_definition(world, session_id="s"):
    """Return the hydrated frozen definition declared by turn start."""

    _, rows, _ = read_logical_state(world, session_id)
    started = next(row.record for row in rows if row.record.kind == "turn_started")
    return started.payload["definition"]
