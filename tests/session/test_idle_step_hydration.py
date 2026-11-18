"""Idle confirmation steps do not rebuild an active logical history."""

from __future__ import annotations

from pathlib import Path
from threading import Thread

from vv_agent.kernel import _boundary_hydration
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_publication import declared_content_ref
from vv_agent.kernel._hydration import reference_digest
from vv_agent.session import _recovery_phase
from vv_agent.session._driver_scope import _Scope
from vv_agent.session.kernel import _Driver, drive
from vv_agent.session.projection import hydrate_record_payload, hydration_for_session_store
from vv_agent.session.records import InboxItem
from vv_agent.types import LLMResponse

from .conftest import open_store
from .helpers import push_fixture_input, wrap_fixture_store
from .test_capacity import runtime, seed_history

SIZE = 20


def _seed_active_prefix(database: Path) -> tuple[Path, str]:
    with open_store(database) as raw:
        raw.install_schema()
        store = wrap_fixture_store(raw)
        turn_id, _records = seed_history(store, database, SIZE)
    return database, turn_id


def _finish_active_prefix(database: Path) -> None:
    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        assert drive(store, "s", runtime=runtime(store, database, []))


def test_finished_active_prefix_does_not_rehydrate_whole_history_on_idle_step(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database, turn_id = _seed_active_prefix(tmp_path / "idle-step.sqlite")
    view_calls: list[tuple[int, str | None]] = []
    batch_sizes: list[int] = []
    read_calls = 0

    original_view = _Driver._logical_view
    original_batch = _boundary_hydration._hydrate_session_records_batch

    def counted_view(self: _Driver):
        view_calls.append((len(self.records), self.fold.state.active_turn_id))
        return original_view(self)

    def counted_batch(sources, hydration, **kwargs):
        batch_sizes.append(len(sources))
        return original_batch(sources, hydration, **kwargs)

    monkeypatch.setattr(_Driver, "_logical_view", counted_view)
    monkeypatch.setattr(_boundary_hydration, "_hydrate_session_records_batch", counted_batch)

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        before_records = store.read("s").records
        hydration = hydration_for_session_store(store, "s")
        completed_before = next(row for row in before_records if row.record.kind == "op_completed")
        payload_before = hydrate_record_payload(completed_before, hydration)
        usage_before = sum(row.record.kind == "usage_observed" for row in before_records)

        selected = runtime(store, database, [])
        assert selected.content_reader is not None
        reader_type = type(selected.content_reader)
        original_read = reader_type.read

        def counted_read(self, *args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            return original_read(self, *args, **kwargs)

        monkeypatch.setattr(reader_type, "read", counted_read)
        assert drive(store, "s", runtime=selected)

        after_records = store.read("s").records
        completed_after = next(row for row in after_records if row.record.kind == "op_completed")
        payload_after = hydrate_record_payload(completed_after, hydration)
        state, _, _ = store.read_state("s")

    assert view_calls == [(SIZE, turn_id)]
    assert batch_sizes == [SIZE]
    assert read_calls > 0
    assert [row.record.encode() for row in after_records[: len(before_records)]] == [
        row.record.encode() for row in before_records
    ]
    assert sum(row.record.kind == "usage_observed" for row in after_records) == usage_before + 1
    assert payload_after == payload_before
    assert state.active_turn_id is None
    assert state.phase == "idle"


def test_finished_history_starts_queued_user_before_follow_up_with_selection(
    tmp_path: Path,
) -> None:
    database, _turn_id = _seed_active_prefix(tmp_path / "queued-inputs.sqlite")
    _finish_active_prefix(database)

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        before = store.read("s").records
        push_fixture_input(store, "s", InboxItem("queued-follow", "follow_up", {"content": "follow"}))
        push_fixture_input(
            store,
            "s",
            InboxItem(
                "queued-selection",
                "model_selection",
                {
                    "selection": {
                        "model": "scripted",
                        "model_settings": {"max_tokens": 321},
                        "route": {},
                    }
                },
            ),
        )
        push_fixture_input(store, "s", InboxItem("queued-user", "user", {"content": "user"}))

        selected = runtime(
            store,
            database,
            [LLMResponse("user answer"), LLMResponse("follow answer")],
        )
        assert drive(store, "s", runtime=selected)
        assert not selected.llm.steps

        records = store.read("s").records
        new_records = records[len(before) :]
        hydration = hydration_for_session_store(store, "s")
        starts = [row for row in new_records if row.record.kind == "turn_started"]
        assert [row.record.payload["input_ids"] for row in starts] == [["queued-user"], ["queued-follow"]]

        applied = {row.record.payload["input"]["input_id"]: row for row in new_records if row.record.kind == "input_applied"}
        assert set(applied) == {"queued-follow", "queued-selection", "queued-user"}
        assert all(row.record.payload["disposition"] == "queued" for key, row in applied.items() if key != "queued-selection")
        assert applied["queued-selection"].record.payload["disposition"] == "applied"

        for input_id in ("queued-user", "queued-follow", "queued-selection"):
            raw_input = applied[input_id].record.payload["input"]
            assert ContentRef.from_dict(raw_input["actor"]["attributes"])
            if input_id != "queued-selection":
                assert declared_content_ref(raw_input["payload"]["content"]) is not None
            hydrated_input = hydrate_record_payload(applied[input_id], hydration)["input"]
            assert isinstance(hydrated_input["actor"]["attributes"], dict)
            if input_id != "queued-selection":
                assert isinstance(hydrated_input["payload"]["content"], str)

        for row in starts:
            raw = row.record.payload
            definition_ref = ContentRef.from_dict(raw["definition"])
            assert definition_ref is not None
            assert ContentRef.from_dict(raw["budget"]) is not None
            hydrated = hydrate_record_payload(row, hydration)
            assert raw["definition_digest"] == reference_digest(definition_ref)
            assert isinstance(hydrated["definition"]["task"], dict)
            assert isinstance(hydrated["budget"], dict)

        plans = [row for row in new_records if row.record.kind == "op_planned" and row.record.payload["purpose"] == "primary"]
        assert len(plans) == 2
        for row in plans:
            request = hydrate_record_payload(row, hydration)["request"]
            assert request["model"] == "scripted"
            assert request["model_settings"]["max_tokens"] == 321


def test_idle_refresh_race_keeps_one_input_and_one_model_call(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database, _turn_id = _seed_active_prefix(tmp_path / "refresh-race.sqlite")
    _finish_active_prefix(database)

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        selected = runtime(store, database, [LLMResponse("raced answer")])
        lease = store.acquire("s", owner="idle-refresh-race", ttl_ms=60_000)
        assert lease is not None
        driver = _Driver(store, "s", selected, _Scope(lease, selected, now_ms=store._now()))
        driver.scope.start()
        assert driver.scope.ready.wait(5)
        original_refresh = driver.refresh
        idle_views: list[int] = []
        original_view = _Driver._logical_view

        def counted_view(self: _Driver):
            if self.fold.state.active_turn_id is None:
                idle_views.append(len(self.records))
            return original_view(self)

        def refresh_then_publish() -> None:
            original_refresh()

            def publish() -> None:
                with open_store(database) as host_raw:
                    host_store = wrap_fixture_store(host_raw)
                    push_fixture_input(host_store, "s", InboxItem("raced", "user", {"content": "raced"}))

            thread = Thread(target=publish, name="idle-host-ingress")
            thread.start()
            thread.join(10)
            assert not thread.is_alive()

        try:
            monkeypatch.setattr(_Driver, "_logical_view", counted_view)
            monkeypatch.setattr(driver, "refresh", refresh_then_publish)
            assert driver.step() is False
            monkeypatch.setattr(driver, "refresh", original_refresh)

            assert driver.step() is True
            assert driver.step() is True

            state, records, _ = store.read_state("s")
            assert state.active_turn_id == "s/turn/raced"
            assert (
                sum(row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "raced" for row in records)
                == 1
            )
            assert sum(row.record.kind == "turn_started" and row.record.payload["input_ids"] == ["raced"] for row in records) == 1
            assert len(selected.llm.steps) == 1
            assert not idle_views
        finally:
            driver.scope.close()
            assert store.release(driver.scope.lease)


def test_idle_compaction_input_starts_the_existing_control_turn(tmp_path: Path, monkeypatch) -> None:
    database, _turn_id = _seed_active_prefix(tmp_path / "compaction-input.sqlite")
    _finish_active_prefix(database)

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        push_fixture_input(
            store,
            "s",
            InboxItem(
                "compact-request",
                "user",
                {
                    "content": {
                        "text": "compact",
                        "metadata": {"request_compaction": True},
                        "app_server_compaction": {},
                    }
                },
            ),
        )
        selected = runtime(store, database, [LLMResponse("compacted")])
        lease = store.acquire("s", owner="idle-compaction-input", ttl_ms=60_000)
        assert lease is not None
        driver = _Driver(store, "s", selected, _Scope(lease, selected, now_ms=store._now()))
        idle_views: list[int] = []
        original_view = _Driver._logical_view

        def counted_view(self: _Driver):
            if self.fold.state.active_turn_id is None:
                idle_views.append(len(self.records))
            return original_view(self)

        try:
            monkeypatch.setattr(_Driver, "_logical_view", counted_view)
            assert driver.step() is True
            assert driver.step() is True

            state, records, _ = store.read_state("s")
            assert state.active_turn_id == "s/turn/compact-request"
            applied = next(
                row
                for row in records
                if row.record.kind == "input_applied" and row.record.payload["input"]["input_id"] == "compact-request"
            )
            assert applied.record.payload["disposition"] == "queued"
            started = next(
                row
                for row in records
                if row.record.kind == "turn_started" and row.record.payload["input_ids"] == ["compact-request"]
            )
            hydration = hydration_for_session_store(store, "s")
            definition = hydrate_record_payload(started, hydration)["definition"]
            assert definition["task"]["metadata"]["_vv_agent_app_server_compaction_only"] is True
            assert len(selected.llm.steps) == 1
            assert not idle_views
        finally:
            driver.scope.close()
            assert store.release(driver.scope.lease)


def test_idle_step_preserves_memory_projection_dispatch(tmp_path: Path, monkeypatch) -> None:
    database, _turn_id = _seed_active_prefix(tmp_path / "memory-dispatch.sqlite")
    _finish_active_prefix(database)
    calls: list[None] = []
    original = _recovery_phase.memory.save_projection

    def counted(driver):
        calls.append(None)
        return original(driver)

    monkeypatch.setattr(_recovery_phase.memory, "save_projection", counted)

    with open_store(database) as raw:
        store = wrap_fixture_store(raw)
        selected = runtime(store, database, [])
        lease = store.acquire("s", owner="idle-memory-projection", ttl_ms=60_000)
        assert lease is not None
        driver = _Driver(store, "s", selected, _Scope(lease, selected, now_ms=store._now()))
        assert driver.step() is False
        assert len(calls) == 1
        assert store.release(lease)
