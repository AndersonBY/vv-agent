from __future__ import annotations

import pytest

from scripts.s1_fixture_values import create_fixture_session

from .scenarios import run_scenario
from .world import InvariantViolation, SimWorld, sqlite_root


def test_world_check_reuses_verified_prefix_without_rescanning(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "cache-hit.sqlite", seed=17) as world:
        run_scenario(world, "tools")
        world.check(quiescent=True)

        observer = world.connection("observer")
        root = sqlite_root(observer)
        queries = []
        actual_rows = root._rows

        def record_query(query, params=()):
            queries.append(query)
            return actual_rows(query, params)

        monkeypatch.setattr(root, "_rows", record_query)
        world.check(quiescent=True)

        assert world._ledger_checks
        assert not any("FROM sk_record" in query or "FROM sk_checkpoint" in query for query in queries)


def test_world_ledger_cache_is_scoped_by_tenant(tmp_path):
    with SimWorld(tmp_path / "tenant-cache.sqlite", seed=17) as world:
        host = world.connection("host")
        tenant_a = host.for_tenant("tenant-a")
        tenant_b = host.for_tenant("tenant-b")
        create_fixture_session(tenant_a, "same", consumers=())
        create_fixture_session(tenant_b, "same", consumers=())

        world.connections["observer"] = tenant_b
        expected_b = world.check_ledger("same")
        world._ledger_checks.clear()

        world.connections["observer"] = tenant_a
        digest_a = world.check_ledger("same")
        world.connections["observer"] = tenant_b
        digest_b = world.check_ledger("same")

        assert digest_a != digest_b
        assert digest_b == expected_b
        assert len(world._ledger_checks) == 2


@pytest.mark.parametrize("tamper", ["record_body", "checkpoint_signature"])
def test_world_check_invalidates_same_head_ledger_tamper(tmp_path, tamper):
    with SimWorld(tmp_path / f"{tamper}.sqlite", seed=17) as world:
        run_scenario(world, "tools")
        world.check(quiescent=True)
        observer = world.connection("observer")
        head = observer._head("s")[0]

        if tamper == "record_body":
            # Same-connection DML must advance total_changes even though data_version does not.
            original = observer._one("SELECT body FROM sk_record WHERE session_id='s' AND seq=2")[0]
            observer._rows("UPDATE sk_record SET body=? WHERE session_id='s' AND seq=2", (original + b" ",))
        else:
            # A different connection is visible through PRAGMA data_version.
            fault = world.connection("fault")
            original = fault._one("SELECT signature FROM sk_checkpoint WHERE session_id='s' AND seq=2")[0]
            forged = bytes([original[0] ^ 1]) + original[1:]
            fault._rows("UPDATE sk_checkpoint SET signature=? WHERE session_id='s' AND seq=2", (forged,))

        assert observer._head("s")[0] == head
        with pytest.raises(InvariantViolation, match="I9"):
            world.check()

        if tamper == "record_body":
            observer._rows("UPDATE sk_record SET body=? WHERE session_id='s' AND seq=2", (original,))
        else:
            fault._rows("UPDATE sk_checkpoint SET signature=? WHERE session_id='s' AND seq=2", (original,))
        world.check(quiescent=True)


def test_world_check_revalidates_cross_connection_inbox_tamper(tmp_path):
    with SimWorld(tmp_path / "inbox-tamper.sqlite", seed=17) as world:
        run_scenario(world, "tools")
        world.check(quiescent=True)
        observer, fault = world.connection("observer"), world.connection("fault")
        head = observer._head("s")[0]

        fault._rows("UPDATE sk_inbox SET consumed_seq=NULL WHERE session_id='s' AND input_id='initial'")

        assert observer._head("s")[0] == head
        with pytest.raises(InvariantViolation, match="I3"):
            world.check()
