#!/usr/bin/env python3
"""Durable session history and explicit retained-turn resume with SQLite."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from vv_agent import Agent, AgentSessionOptions, InteractiveAgentClient, SQLiteStore, VvLlmModelProvider


def main() -> None:
    with TemporaryDirectory(prefix="vv-agent-example-") as temporary_workspace:
        workspace = Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", temporary_workspace)).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        db_path = Path(os.getenv("VV_AGENT_EXAMPLE_DB", str(workspace / ".vv-agent-state" / "sessions.db")))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with SQLiteStore.standalone(db_path) as store:
            if not store.connection.execute("PRAGMA user_version").fetchone()[0]:
                store.install_schema()
            client = InteractiveAgentClient(
                options=AgentSessionOptions(
                    model_provider=VvLlmModelProvider(
                        settings_file=Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py")),
                        default_backend=os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot"),
                    ),
                    workspace=workspace,
                    session_store=store,
                )
            )
            session = client.create_session(
                agent=Agent(
                    "durable-demo", "Complete the requested task carefully.", model=os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3")
                ),
                session_id=os.getenv("VV_AGENT_EXAMPLE_SESSION_ID", "example-21"),
            )
            try:
                state, _, _ = store.read_state(session.session_id)
                if state.active_turn_id is not None:
                    result = session.continue_run()
                else:
                    result = session.prompt(os.getenv("VV_AGENT_EXAMPLE_PROMPT", "Calculate 2+3 and finish."))
                print(result.raw_result.session_id, result.raw_result.turn_id, result.status.value, result.final_output)
            finally:
                client.driver.close()


if __name__ == "__main__":
    main()
