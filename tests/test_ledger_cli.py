from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from session.helpers import item
from support.s1_fixture_values import (
    create_fixture_session,
    fixture_key_provider,
    push_fixture_input,
    wrap_fixture_store,
)

import vv_agent.cli as cli
from vv_agent.kernel._store import DEFAULT_TENANT_ID
from vv_agent.session.sqlite import SQLiteStore


class _Signer:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.key_id = "ledger/test-key"

    def bind_key(self):
        signer = self

        class Bound:
            key_id = signer.key_id

            @staticmethod
            def sign(message: bytes) -> bytes:
                return signer.key.sign(message)

        return Bound()

    def verify(self, key_id: str, message: bytes, signature: bytes) -> bool:
        if key_id != self.key_id:
            return False
        try:
            self.key.public_key().verify(signature, message)
        except InvalidSignature:
            return False
        return True


def _create_hash_only_session(path: Path, session_id: str = "session-1") -> tuple[int, str]:
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(allow_unsigned=True)
        create_fixture_session(wrap_fixture_store(store, fixture_key_provider()), session_id)
        head = store._head(session_id)[0]
        return head, store._chain_anchor(session_id, head)


def _create_signed_session(path: Path, session_id: str = "session-1", *, tenant_id: str = DEFAULT_TENANT_ID) -> None:
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(_Signer())
        tenant_store = store if tenant_id == DEFAULT_TENANT_ID else store.for_tenant(tenant_id)
        create_fixture_session(wrap_fixture_store(tenant_store, fixture_key_provider()), session_id)
        assert (
            store._one(
                "SELECT count(*) FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s",
                (tenant_id, session_id),
            )[0]
            > 0
        )


def _leave_only_admission_proof(path: Path, session_id: str = "session-1") -> None:
    with SQLiteStore.standalone(str(path)) as store:
        store.install_schema()
        store.configure_input_admission(_Signer())
        wrapped = wrap_fixture_store(store, fixture_key_provider())
        create_fixture_session(wrapped, session_id)
        push_fixture_input(wrapped, session_id, item("input-1"))
        store.connection.execute(
            "DELETE FROM sk_checkpoint WHERE tenant_id=? AND session_id=?",
            (store.tenant_id, session_id),
        )
        assert (
            store._one("SELECT count(*) FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s", (store.tenant_id, session_id))[
                0
            ]
            == 0
        )
        assert (
            store._one(
                "SELECT count(*) FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND admission_signature IS NOT NULL",
                (store.tenant_id, session_id),
            )[0]
            > 0
        )


def _run_verify(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, dict, str]:
    code = cli._main(["ledger", "verify", *args])
    captured = capsys.readouterr()
    assert captured.err == ""
    return code, json.loads(captured.out), captured.out


def test_hash_only_sqlite_session_reports_observable_scope_and_anchor(tmp_path, capsys) -> None:
    path = tmp_path / "sessions.sqlite"
    head_seq, chain_digest = _create_hash_only_session(path)

    code, payload, output = _run_verify(
        capsys,
        "--sqlite",
        str(path),
        "--session",
        "session-1",
        "--allow-unsigned",
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["backend"] == "sqlite"
    assert payload["checked_scope"] == {
        "kind": "session",
        "tenant_id": DEFAULT_TENANT_ID,
        "session_id": "session-1",
        "session_count": 1,
        "sessions_verified": 1,
    }
    assert payload["observed_heads"] == [
        {"session_id": "session-1", "seq": head_seq, "digest": chain_digest},
    ]
    assert payload["authenticity"] == "hash-only"
    assert payload["expected_head_present"] is False
    assert payload["suffix_completeness"] is False
    assert payload["error"] is None
    assert str(path) not in output

    code, payload, _output = _run_verify(
        capsys,
        "--sqlite",
        str(path),
        "--session",
        "session-1",
        "--allow-unsigned",
        "--expected-head",
        f"{head_seq},{chain_digest}",
    )

    assert code == 0
    assert payload["expected_head_present"] is True
    assert payload["suffix_completeness"] is True


def test_expected_head_mismatch_reports_observed_head_and_nonzero(tmp_path, capsys) -> None:
    path = tmp_path / "sessions.sqlite"
    head_seq, chain_digest = _create_hash_only_session(path)

    code, payload, _output = _run_verify(
        capsys,
        "--sqlite",
        str(path),
        "--session",
        "session-1",
        "--allow-unsigned",
        "--expected-head",
        f"{head_seq + 1},{chain_digest}",
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["observed_heads"] == [
        {"session_id": "session-1", "seq": head_seq, "digest": chain_digest},
    ]
    assert payload["checked_scope"]["sessions_verified"] == 0
    assert payload["expected_head_present"] is True
    assert payload["suffix_completeness"] is False
    assert payload["error"] == {
        "code": "expected_head_mismatch",
        "message": f"ledger head does not match expected_head at seq {head_seq}",
    }


def test_ledger_integrity_failure_reports_the_store_first_failure(tmp_path, capsys) -> None:
    path = tmp_path / "sessions.sqlite"
    head_seq, _chain_digest = _create_hash_only_session(path)
    with SQLiteStore.standalone(str(path)) as store:
        store.connection.execute("UPDATE sk_record SET chain_digest=? WHERE seq=?", ("0" * 64, head_seq))

    code, payload, _output = _run_verify(
        capsys,
        "--sqlite",
        str(path),
        "--session",
        "session-1",
        "--allow-unsigned",
    )

    assert code == 1
    assert payload["error"] == {
        "code": "ledger_integrity_error",
        "message": f"stored record chain digest mismatch at seq {head_seq}",
    }


@pytest.mark.parametrize("evidence", ["checkpoint", "admission-proof"])
def test_signed_evidence_requires_host_verifier_before_hash_only_mode(tmp_path, capsys, monkeypatch, evidence) -> None:
    path = tmp_path / "signed.sqlite"
    if evidence == "checkpoint":
        _create_signed_session(path)
    else:
        _leave_only_admission_proof(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("hash-only mode must not be configured or invoked")

    monkeypatch.setattr(SQLiteStore, "configure_input_admission", forbidden)
    monkeypatch.setattr(SQLiteStore, "verify_ledger", forbidden)

    code, payload, _output = _run_verify(
        capsys,
        "--sqlite",
        str(path),
        "--session",
        "session-1",
        "--allow-unsigned",
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["authenticity"] == "signed"
    assert payload["checked_scope"]["sessions_verified"] == 0
    assert payload["observed_heads"] == []
    assert payload["error"] == {
        "code": "signed_evidence_requires_host_verifier",
        "message": (
            "signed ledger evidence was detected; a host-configured verifier is required and hash-only "
            "verification was not performed"
        ),
    }


def test_tenant_hash_only_refuses_when_any_target_session_has_signed_evidence(tmp_path, capsys, monkeypatch) -> None:
    path = tmp_path / "mixed.sqlite"
    _create_hash_only_session(path, "hash-only-session")
    _create_signed_session(path, "signed-session")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("hash-only mode must not be configured or invoked")

    monkeypatch.setattr(SQLiteStore, "configure_input_admission", forbidden)
    monkeypatch.setattr(SQLiteStore, "verify_ledger", forbidden)

    code, payload, _output = _run_verify(
        capsys,
        "--sqlite",
        str(path),
        "--tenant",
        DEFAULT_TENANT_ID,
        "--allow-unsigned",
    )

    assert code == 1
    assert payload["checked_scope"]["session_count"] == 2
    assert payload["checked_scope"]["sessions_verified"] == 0
    assert payload["error"]["code"] == "signed_evidence_requires_host_verifier"


def test_signed_mode_fails_closed_without_a_host_configured_verifier(tmp_path, capsys) -> None:
    path = tmp_path / "does-not-exist.sqlite"

    code, payload, _output = _run_verify(capsys, "--sqlite", str(path), "--session", "session-1")

    assert code == 1
    assert payload["ok"] is False
    assert payload["authenticity"] == "signed"
    assert payload["observed_heads"] == []
    assert payload["error"] == {
        "code": "unconfigured_verifier",
        "message": (
            "signed ledger verification requires a host-configured verifier; this CLI does not load verifier "
            "or KMS credentials. Use --allow-unsigned only for local SQLite hash-only verification."
        ),
    }
    assert not path.exists()


def test_tenant_scope_does_not_enumerate_other_tenants(tmp_path, capsys) -> None:
    path = tmp_path / "sessions.sqlite"
    _create_hash_only_session(path, "default-session")
    _create_signed_session(path, "other-session", tenant_id="other-tenant")

    code, payload, _output = _run_verify(
        capsys,
        "--sqlite",
        str(path),
        "--tenant",
        DEFAULT_TENANT_ID,
        "--allow-unsigned",
    )

    assert code == 0
    assert payload["checked_scope"] == {
        "kind": "tenant",
        "tenant_id": DEFAULT_TENANT_ID,
        "session_count": 1,
        "sessions_verified": 1,
    }
    assert [head["session_id"] for head in payload["observed_heads"]] == ["default-session"]


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (("--session", "one", "--tenant", "two"), "not allowed with argument --session"),
        (
            ("--tenant", "one", "--expected-head", f"1,{'0' * 64}"),
            "--expected-head requires --session and cannot be used with --tenant",
        ),
        (("--session", "one", "--expected-head", "1,not-a-digest"), "digest must be 64 lowercase"),
    ],
)
def test_conflicting_or_malformed_arguments_fail_during_parsing(tmp_path, capsys, extra_args, message) -> None:
    code = cli._main(
        [
            "ledger",
            "verify",
            "--sqlite",
            str(tmp_path / "sessions.sqlite"),
            "--allow-unsigned",
            *extra_args,
        ]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert message in captured.err
