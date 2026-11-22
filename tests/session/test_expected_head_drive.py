from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path

import pytest

from vv_agent.agent import Agent
from vv_agent.config import ResolvedModelConfig
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentBindingMismatch
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.run_config import RunConfig
from vv_agent.session.kernel import Runtime, drive, read_state
from vv_agent.session.records import InboxItem
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse

from .helpers import (
    create_fixture_session,
    fixture_content_reader,
    push_fixture_input,
    record,
    wrap_fixture_store,
)

RESOLVED = ResolvedModelConfig("scripted", "m", "m", "m", [])


def _runtime(store, tmp_path: Path, steps) -> Runtime:
    return Runtime(
        Agent("expected-head", "Be precise."),
        RunConfig(workspace=tmp_path),
        RESOLVED,
        ScriptedLLM(steps),
        lambda: nullcontext(store),
        content_reader=fixture_content_reader(store, "s"),
    )


def _entry(entry: str, runtime: Runtime) -> Runtime | Callable[[str], Runtime]:
    if entry == "runtime":
        return runtime

    def factory(_session_id: str) -> Runtime:
        return runtime

    return factory


def _forge_last_hash_only_record(store: SQLiteStore) -> str:
    seq = store._head("s")[0]
    body, commit_id = store._one("SELECT body,commit_id FROM sk_record WHERE seq=?", (seq,))
    value = json.loads(body)
    value["payload"]["handler_version"] = "forged"
    forged_body = canonical_json_bytes(value)
    forged_digest = sha256(forged_body).hexdigest()
    previous_chain = store._one("SELECT chain_digest FROM sk_record WHERE seq=?", (seq - 1,))[0]
    forged_chain = sha256(bytes.fromhex(previous_chain) + seq.to_bytes(8, "big") + bytes.fromhex(forged_digest)).hexdigest()

    manifest = json.loads(
        store._one("SELECT body FROM sk_commit WHERE tenant_id='default' AND session_id='s' AND commit_id=?", (commit_id,))[0]
    )
    manifest["records"][0]["digest"] = forged_digest
    forged_manifest = canonical_json_bytes(manifest)
    forged_commit_digest = sha256(forged_manifest).hexdigest()
    store.connection.execute(
        "UPDATE sk_record SET body=?,digest=?,chain_digest=?,commit_digest=? WHERE session_id='s' AND seq=?",
        (forged_body, forged_digest, forged_chain, forged_commit_digest, seq),
    )
    store.connection.execute(
        "UPDATE sk_commit SET body=?,digest=? WHERE tenant_id='default' AND session_id='s' AND commit_id=?",
        (forged_manifest, forged_commit_digest, commit_id),
    )
    return forged_chain


@pytest.mark.parametrize("entry", ["runtime", "factory"])
def test_drive_expected_head_rejects_rehashed_rollback_before_model(tmp_path: Path, entry: str) -> None:
    with SQLiteStore.standalone(str(tmp_path / f"tampered-{entry}.sqlite")) as raw:
        raw.install_schema()
        raw.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw)
        create_fixture_session(store, "s")
        lease = store.acquire("s", owner="test", ttl_ms=10_000)
        assert lease is not None
        with store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=1,
                commit_id="trusted-tail",
                records=(record("turn_started", sid="s"),),
            )
        trusted_head = store.verify_ledger("s")
        assert store.release(lease)
        push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))

        forged_chain = _forge_last_hash_only_record(raw)
        assert store.verify_ledger("s") == (2, forged_chain)
        calls: list[object] = []
        runtime = _runtime(
            store,
            tmp_path,
            [lambda request: calls.append(request) or LLMResponse("must not run")],
        )

        with pytest.raises(ContentBindingMismatch, match="expected_head"):
            drive(store, "s", runtime=_entry(entry, runtime), expected_head=trusted_head)

        assert calls == []
        assert store.read("s").head_seq == 2
        assert read_state(store, "s")[0].applied_inputs == {}


@pytest.mark.parametrize("entry", ["runtime", "factory"])
def test_drive_expected_head_advances_across_cycles(tmp_path: Path, entry: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with SQLiteStore.standalone(str(tmp_path / f"intact-{entry}.sqlite")) as raw:
        raw.install_schema()
        raw.configure_input_admission(allow_unsigned=True)
        store = wrap_fixture_store(raw)
        create_fixture_session(store, "s")
        trusted_head = store.verify_ledger("s")
        verify_ledger = store.verify_ledger
        checked_heads = []

        def verify_anchored(session_id, *, expected_head=None):
            assert expected_head is not None
            result = verify_ledger(session_id, expected_head=expected_head)
            checked_heads.append(result)
            return result

        monkeypatch.setattr(store, "verify_ledger", verify_anchored)
        push_fixture_input(store, "s", InboxItem("initial", "user", {"content": "go"}))
        calls: list[object] = []
        runtime = _runtime(
            store,
            tmp_path,
            [lambda request: calls.append(request) or LLMResponse("done")],
        )

        assert drive(store, "s", runtime=_entry(entry, runtime), expected_head=trusted_head)

        assert len(calls) == 1
        state, records, _ = read_state(store, "s")
        assert state.active_turn_id is None
        assert [stored.record.kind for stored in records].count("turn_ended") == 1
        assert checked_heads[-1] == verify_ledger("s") != trusted_head
