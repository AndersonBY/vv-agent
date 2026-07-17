#!/usr/bin/env python3
"""Durable checkpoint v2: resume or replay one stable Runner task."""

from __future__ import annotations

import os
from pathlib import Path

from vv_agent import Agent, CheckpointConfig, RunConfig, Runner
from vv_agent.checkpoint import ResumePolicy
from vv_agent.runtime.stores.sqlite import SqliteStateStore


def main() -> None:
    settings_file = Path(os.getenv("V_AGENT_LOCAL_SETTINGS", "local_settings.py"))
    backend = os.getenv("V_AGENT_EXAMPLE_BACKEND", "moonshot")
    model = os.getenv("V_AGENT_EXAMPLE_MODEL", "kimi-k2.6")
    workspace = Path(os.getenv("V_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    db_path = Path(
        os.getenv(
            "V_AGENT_EXAMPLE_DB",
            str(workspace / ".vv-agent-state" / "checkpoints.db"),
        )
    ).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_key = os.getenv(
        "V_AGENT_EXAMPLE_CHECKPOINT_KEY",
        "example-21-state-checkpoint",
    )
    prompt = os.getenv(
        "V_AGENT_EXAMPLE_PROMPT",
        "Calculate 2+3, briefly verify the result, and finish.",
    )

    store = SqliteStateStore(db_path)
    config = RunConfig(
        model=model,
        settings_file=settings_file,
        default_backend=backend,
        workspace=workspace,
        max_cycles=5,
        checkpoint_config=CheckpointConfig(
            key=checkpoint_key,
            resume_policy=ResumePolicy.RESUME_IF_PRESENT,
            store=store,
            capability_refs={
                "workspace": {"id": "workspace.example-21", "version": "1"},
            },
        ),
    )
    agent = Agent(
        name="checkpoint-demo",
        instructions=(
            "Complete the requested task carefully. Use the finish tool only after "
            "the answer is ready."
        ),
        model=model,
    )

    print(f"[demo] checkpoint={checkpoint_key}")
    print(f"[demo] database={db_path}")
    result = Runner.run_sync(agent, prompt, run_config=config)
    print(f"[demo] status={result.status.value}")
    print(f"[demo] output={result.final_output}")

    retained = store.load_checkpoint_v2(checkpoint_key)
    if retained is not None:
        print(
            "[demo] durable_state="
            f"cycle:{retained.cycle_index} "
            f"resume_attempt:{retained.resume_attempt} "
            f"terminal_acknowledged:{retained.terminal_acknowledged}"
        )
    print("[demo] Run the same command again to replay or resume this checkpoint.")
    store.close()


if __name__ == "__main__":
    main()
