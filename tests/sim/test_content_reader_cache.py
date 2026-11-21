"""Fail-closed cache behavior for the real SimWorld content reader."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from vv_agent.kernel._boundary_hydration import hydrate_session_records
from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._hydration import IntegrityFailure, KeyUnavailable, MissingContentLink

from .assembly import SimContentReader
from .scenarios import run_scenario
from .world import SimWorld


def _linked_record(store, session_id: str = "s"):
    row = store._one(
        "SELECT l.record_seq,l.slot,c.content_id,c.mac,c.size,c.key_id,c.labels "
        "FROM sk_content_link l JOIN sk_content c ON c.tenant_id=l.tenant_id "
        "AND c.session_id=l.session_id AND c.content_id=l.content_id "
        "WHERE l.tenant_id=%s AND l.session_id=%s AND l.record_seq IS NOT NULL "
        "ORDER BY l.record_seq,l.slot LIMIT 1",
        (store.tenant_id, session_id),
    )
    assert row is not None
    record_seq, slot, content_id, mac, size, key_id, labels = row
    ref = ContentRef.from_dict(
        {
            "content_id": content_id,
            "mac": mac,
            "size": size,
            "key_id": key_id,
            "labels": json.loads(labels),
        }
    )
    return store.tenant_id, f"record:{record_seq}", slot, ref


def _tamper_cached_content(mutator, tenant_id: str, session_id: str, ref: ContentRef, field: str) -> None:
    row = mutator._one(
        "SELECT nonce,ciphertext,object_locator FROM sk_content WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
        (tenant_id, session_id, ref.content_id),
    )
    assert row is not None
    nonce, ciphertext, object_locator = (None if value is None else bytes(value) for value in row)
    with mutator.atomic():
        if field == "ciphertext":
            assert ciphertext is not None
            mutator._rows(
                "UPDATE sk_content SET ciphertext=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (bytes([ciphertext[0] ^ 1]) + ciphertext[1:], tenant_id, session_id, ref.content_id),
            )
        elif field == "nonce":
            assert nonce is not None
            mutator._rows(
                "UPDATE sk_content SET nonce=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                (bytes([nonce[0] ^ 1]) + nonce[1:], tenant_id, session_id, ref.content_id),
            )
        elif field == "object_locator":
            mutator._rows(
                "UPDATE sk_content SET ciphertext=NULL,object_locator=%s WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                ((object_locator or b"sim-content-object") + b"/tampered", tenant_id, session_id, ref.content_id),
            )
        elif field == "wrapped_key":
            row = mutator._one(
                "SELECT wrapped_key FROM sk_session_key WHERE tenant_id=%s AND session_id=%s",
                (tenant_id, session_id),
            )
            assert row is not None
            wrapped_key = bytes(row[0])
            mutator._rows(
                "UPDATE sk_session_key SET wrapped_key=%s WHERE tenant_id=%s AND session_id=%s",
                (bytes([wrapped_key[0] ^ 1]) + wrapped_key[1:], tenant_id, session_id),
            )
        else:
            raise AssertionError(f"unknown mutation: {field}")


def test_repeated_read_and_boundary_frame_reuse_validated_plaintext(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "content-cache.sqlite", seed=17) as world:
        run_scenario(world, "tools")
        store = world.connection("observer")
        _state, rows, _ = store.read_state("s")
        hydration = world.hydration("s")
        reader = hydration.reader
        tenant_id, owner, slot, ref = _linked_record(store)
        world._content_bodies.clear()

        before = world.content_cipher.decrypts
        body = reader.read(tenant_id, "s", owner, slot, ref)
        after_first = world.content_cipher.decrypts
        assert after_first == before + 1
        assert reader.read(tenant_id, "s", owner, slot, ref) == body
        assert world.content_cipher.decrypts == after_first
        world.create("other")
        after_other_commit = world.content_cipher.decrypts
        assert reader.read(tenant_id, "s", owner, slot, ref) == body
        assert world.content_cipher.decrypts == after_other_commit

        actual_frame = reader._reader._read_frame
        frames = []

        @contextmanager
        def observed_frame(**kwargs):
            frames.append(kwargs)
            with actual_frame(**kwargs) as framed:
                yield framed

        monkeypatch.setattr(reader._reader, "_read_frame", observed_frame)
        hydrate_session_records(rows, hydration)
        after_boundary = world.content_cipher.decrypts
        hydrate_session_records(rows, hydration)
        assert len(frames) == 2
        assert world.content_cipher.decrypts == after_boundary


def test_unchanged_sqlite_and_provider_revision_uses_validated_cache(tmp_path, monkeypatch):
    with SimWorld(tmp_path / "content-cache-version.sqlite", seed=17) as world:
        run_scenario(world, "tools")
        store = world.connection("observer")
        hydration = world.hydration("s")
        reader = hydration.reader
        tenant_id, owner, slot, ref = _linked_record(store)
        world._content_bodies.clear()

        body = reader.read(tenant_id, "s", owner, slot, ref)
        cache_key = (tenant_id, "s", owner, slot, ref)
        assert world._content_bodies[cache_key][2] is not None
        decrypts = world.content_cipher.decrypts

        def unexpected_validation(*_args, **_kwargs):
            raise AssertionError("unchanged cache entry ran full validation")

        monkeypatch.setattr(SimContentReader, "_assert_cached_link", staticmethod(unexpected_validation))
        monkeypatch.setattr(SimContentReader, "_content_state", staticmethod(unexpected_validation))
        monkeypatch.setattr(SimContentReader, "_assert_provider_key", unexpected_validation)

        assert reader.read(tenant_id, "s", owner, slot, ref) == body
        with hydration.reader._reader._read_frame(record_seqs=(int(owner.removeprefix("record:")),)) as framed:
            assert reader._read(framed, tenant_id, "s", owner, slot, ref) == body
        assert world.content_cipher.decrypts == decrypts


@pytest.mark.parametrize("mutation", ["link", "gc_pending"])
def test_cross_connection_content_invalidation_rejects_old_plaintext(tmp_path, mutation):
    with SimWorld(tmp_path / f"content-{mutation}.sqlite", seed=17) as world:
        run_scenario(world, "tools")
        store = world.connection("observer")
        reader = world.hydration("s").reader
        tenant_id, owner, slot, ref = _linked_record(store)
        world._content_bodies.clear()
        cached = reader.read(tenant_id, "s", owner, slot, ref)
        decrypts = world.content_cipher.decrypts
        assert reader.read(tenant_id, "s", owner, slot, ref) == cached
        assert world.content_cipher.decrypts == decrypts

        record_seq = int(owner.removeprefix("record:"))
        mutator = world.connection("mutator")
        with mutator.atomic():
            if mutation == "link":
                mutator._rows(
                    "DELETE FROM sk_content_link WHERE tenant_id=%s AND session_id=%s AND record_seq=%s AND slot=%s",
                    (tenant_id, "s", record_seq, slot),
                )
            else:
                mutator._rows(
                    "UPDATE sk_content SET state='gc_pending' WHERE tenant_id=%s AND session_id=%s AND content_id=%s",
                    (tenant_id, "s", ref.content_id),
                )

        with pytest.raises(MissingContentLink):
            reader.read(tenant_id, "s", owner, slot, ref)
        assert world.content_cipher.decrypts == decrypts


@pytest.mark.parametrize("mutation", ["ciphertext", "nonce", "object_locator", "wrapped_key"])
def test_cross_connection_state_tamper_rejects_cached_plaintext(tmp_path, mutation):
    with SimWorld(tmp_path / f"content-cache-{mutation}.sqlite", seed=17) as world:
        run_scenario(world, "tools")
        store = world.connection("observer")
        reader = world.hydration("s").reader
        tenant_id, owner, slot, ref = _linked_record(store)
        world._content_bodies.clear()
        cached = reader.read(tenant_id, "s", owner, slot, ref)
        decrypts = world.content_cipher.decrypts
        assert reader.read(tenant_id, "s", owner, slot, ref) == cached

        _tamper_cached_content(world.connection("mutator"), tenant_id, "s", ref, mutation)

        with pytest.raises(IntegrityFailure):
            reader.read(tenant_id, "s", owner, slot, ref)
        assert world.content_cipher.decrypts == decrypts
        assert (tenant_id, "s", owner, slot, ref) not in world._content_bodies


def test_provider_destroy_invalidates_cached_plaintext(tmp_path):
    with SimWorld(tmp_path / "content-provider-destroy.sqlite", seed=17) as world:
        run_scenario(world, "tools")
        store = world.connection("observer")
        reader = world.hydration("s").reader
        tenant_id, owner, slot, ref = _linked_record(store)
        world._content_bodies.clear()
        cached = reader.read(tenant_id, "s", owner, slot, ref)
        decrypts = world.content_cipher.decrypts
        assert reader.read(tenant_id, "s", owner, slot, ref) == cached

        revision = world.content_provider.revision
        destroyed = world.content_provider.destroy(tenant_id, ref.key_id)
        assert destroyed.accepted
        assert world.content_provider.revision == revision + 1
        repeated = world.content_provider.destroy(tenant_id, ref.key_id)
        assert repeated.accepted
        assert world.content_provider.revision == revision + 1

        with pytest.raises(KeyUnavailable):
            reader.read(tenant_id, "s", owner, slot, ref)
        assert world.content_cipher.decrypts == decrypts
        assert (tenant_id, "s", owner, slot, ref) not in world._content_bodies


def test_worker_compatibility_claims_do_not_decrypt_content(tmp_path):
    with SimWorld(tmp_path / "worker-compatibility.sqlite", seed=17) as world:
        run_scenario(world, "worker_compatibility")
        claims = [claim for claim in world.worker_claims if "decrypts_before" in claim]
        assert claims
        assert all(claim["decrypts_before"] == claim["decrypts_after"] for claim in claims)
