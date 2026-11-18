from pathlib import Path

from vv_agent.agent import Agent
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._store import ContentSessionTx
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.run_config import RunConfig
from vv_agent.session._drive_phase import _KeyUnavailableTrackingReader
from vv_agent.session.content_runtime import wrap_session_content_store
from vv_agent.session.kernel import Runtime, drive, read_state
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

from .conftest import open_store
from .helpers import create_fixture_session, fixture_key_provider, push_fixture_input

SESSION = "content-tx-regression"


def test_drive_binds_explicit_content_capability_without_mutating_raw_store(tmp_path: Path) -> None:
    database = tmp_path / "content-tx-regression.sqlite"
    with SQLiteStore.standalone(str(database)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        wrapped = wrap_session_content_store(store, fixture_key_provider())
        create_fixture_session(wrapped, SESSION)
        push_fixture_input(wrapped, SESSION, InboxItem("initial", "user", {"content": "go"}))

        with store.atomic() as tx:
            assert not isinstance(tx, ContentSessionTx)

        runtime = Runtime(
            agent=Agent(name="kernel", instructions="Be precise."),
            config=RunConfig(workspace=tmp_path, session_memory_enabled=False),
            resolved=ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", []),
            llm=ScriptedLLM([LLMResponse("done")]),
            heartbeat_store=lambda: open_store(database),
        )
        drive(store, SESSION, runtime=runtime)

        state, records, _ = read_state(store, SESSION)
        assert state.active_turn_id is None
        assert [record.record.payload["status"] for record in records if record.record.kind == "turn_ended"] == ["completed"]

        tracked_reader = runtime.content_reader
        assert isinstance(tracked_reader, _KeyUnavailableTrackingReader)
        reader = tracked_reader.reader
        assert reader.binding == (store.tenant_id, SESSION)
        drive(store, SESSION, runtime=runtime)
        assert runtime.content_reader is tracked_reader
        assert tracked_reader.reader is reader

        with store.atomic() as tx:
            assert not isinstance(tx, ContentSessionTx)
