import json
from hashlib import sha256
from unittest.mock import patch

import pytest

from scripts import session_kernel_benchmark as benchmark


def test_help_exposes_reproducible_a12_controls(capsys):
    with pytest.raises(SystemExit) as stopped:
        benchmark.parse_args(["--help"])

    assert stopped.value.code == 0
    output = capsys.readouterr().out
    for option in ("--turns", "--history-mb-per-turn", "--cold-mode", "--baseline-json"):
        assert option in output


def test_a12_uses_complete_history_for_every_turn():
    target = 10 * benchmark.MIB

    message, logical_bytes, content_chars = benchmark.a12_message(target)

    assert message == {"role": "assistant", "content": "x" * content_chars}
    assert len(message["content"].encode()) == content_chars
    assert logical_bytes == len(benchmark.canonical_json_bytes(message))
    assert logical_bytes == target


def test_legacy_selector_does_not_silently_add_other_categories():
    history_only = benchmark.parse_args(["--sizes", "100", "--samples", "1"])
    catalog_only = benchmark.parse_args(["--sessions", "100", "--samples", "1"])
    a12 = benchmark.parse_args(["--turns", "100", "--history-mb-per-turn", "10"])

    assert history_only.sizes == [100]
    assert history_only.sessions == []
    assert catalog_only.sizes == []
    assert catalog_only.sessions == [100]
    assert a12.sizes == []
    assert a12.sessions == []
    assert a12.a12_requested


def test_a12_interface_rejects_partial_or_unscoped_cold_configuration():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--turns", "100"])
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--turns",
                "100",
                "--history-mb-per-turn",
                "10",
                "--cold-mode",
                "restart",
            ]
        )


def _code(revision="current", *, dirty=False):
    fingerprint = sha256(revision.encode()).hexdigest()
    return {
        "revision": revision,
        "dirty": dirty,
        "fingerprint": fingerprint,
        "dirty_fingerprint": sha256(f"dirty:{revision}".encode()).hexdigest() if dirty else None,
        "source_status": ["M tracker"] if dirty else [],
        "script_sha256": fingerprint,
    }


def _timing(value):
    return benchmark.timing([value])


def _payload(
    *,
    revision="current",
    protocol="S1",
    append_ms=40.0,
    reuse=100,
    record_total=737280,
    cold_mode="none",
    cold_samples=0,
):
    return {
        "revision": revision,
        "code": _code(revision),
        "protocol": protocol,
        "postgres": {
            "version": "PostgreSQL 18.6",
            "config": {"shared_buffers": "128MB", "fsync": "on"},
        },
        "samples": 2,
        "cold_mode": cold_mode,
        "cold_samples": cold_samples,
        "workload": {
            "protocol": protocol,
            "semantic": "one fixed historical message referenced in full by every turn",
            "completed_turns": 100,
            "active_turn_after_history": 1,
            "target_history_bytes_per_turn": 10 * benchmark.MIB,
            "logical_records": 209,
            "logical_history_bytes_per_turn": 10 * benchmark.MIB,
            "logical_history_bytes_total": 100 * 10 * benchmark.MIB,
            "message_plaintext_bytes_each": 10 * benchmark.MIB,
            "message_string_bytes_each": 10 * benchmark.MIB - 35,
            "message_content_chars_each": 10 * benchmark.MIB - 35,
            "message_links_expected": 100,
            "unique_messages_expected": 1,
            "content_dedup_key": "tenant/session/key_id/HMAC/labels",
            "message": "one identical assistant JSON message nested in every historical turn definition",
        },
        "timings": {
            "warm_cached_read": _timing(append_ms),
            "warm_uncached_read": _timing(append_ms),
            "warm_fold": _timing(append_ms),
            "warm_append": _timing(append_ms),
            "warm_drive": _timing(append_ms),
        },
        "bytes_and_reuse": {
            "sql_records": {
                "table": {
                    "table_bytes_including_toast": record_total,
                    "total_bytes": record_total,
                }
            },
            "content_store": {
                "table": {"table_bytes_including_toast": 1, "total_bytes": 1},
                "relations": {"sk_content_link_total_bytes": 1},
            },
            "message_reuse": {"reuse_ratio": reuse},
        },
    }


def test_baseline_comparison_reports_workload_timing_and_reuse_regressions():
    baseline = _payload(revision="baseline", append_ms=40.0)
    current = _payload(revision="current", append_ms=50.0, reuse=50)

    comparison = benchmark.compare_a12_baseline(current, baseline, 1.0)

    assert not comparison["passed"]
    assert not comparison["time_checks"]["warm_append"]["median_not_greater"]
    assert not comparison["message_reuse"]["current_not_lower"]
    assert "timing regression: warm_append" in comparison["failures"]


def test_nested_baseline_keeps_outer_revision_and_excludes_cold_diagnostics():
    baseline = _payload(
        revision="baseline",
        cold_mode="restart",
        cold_samples=2,
    )
    current = _payload(
        revision="current",
        cold_mode="restart",
        cold_samples=2,
    )
    for payload in (baseline, current):
        payload["timings"]["cold_read"] = _timing(20.0)
        payload["timings"]["cold_drive"] = _timing(30.0)
        payload["timings"]["postgres_restart_ms"] = [100.0, 110.0]
        payload["timings"]["warm_drive_ms"] = 30.0
    nested = {
        "revision": "nested-should-not-win",
        "workload": baseline.pop("workload"),
        "timings": baseline.pop("timings"),
        "bytes_and_reuse": baseline.pop("bytes_and_reuse"),
    }
    baseline["a12"] = nested

    comparison = benchmark.compare_a12_baseline(current, baseline, 1.0)

    assert comparison["passed"]
    assert comparison["baseline_revision"] == "baseline"
    assert "postgres_restart_ms" not in comparison["time_checks"]
    assert "warm_drive_ms" not in comparison["time_checks"]
    assert comparison["diagnostics"]["postgres_restart_ms"] == {
        "baseline": [100.0, 110.0],
        "current": [100.0, 110.0],
    }


def test_comparator_rejects_protocol_and_incomplete_dirty_evidence():
    protocol_baseline = _payload(revision="baseline", protocol="S2")
    protocol_comparison = benchmark.compare_a12_baseline(
        _payload(revision="current"),
        protocol_baseline,
        1.0,
    )

    assert not protocol_comparison["checks"]["comparable_protocol"]
    assert protocol_comparison["time_checks"] == {}
    assert "incomparable protocol" in protocol_comparison["failures"]

    dirty_current = _payload(revision="current")
    dirty_current["code"] = _code("current", dirty=True)
    dirty_current["code"]["dirty_fingerprint"] = None
    dirty_comparison = benchmark.compare_a12_baseline(
        dirty_current,
        _payload(revision="baseline"),
        1.0,
    )

    assert not dirty_comparison["checks"]["code_provenance_complete"]
    assert dirty_comparison["time_checks"] == {}
    assert "incomparable code revision/dirty fingerprint" in dirty_comparison["failures"]


def test_successful_comparison_is_printed_to_stdout(capsys):
    payload = {
        "comparison": {"passed": True, "marker": "visible-on-success"},
        "hard_gate": {"passed": True},
    }
    with patch.object(benchmark, "run", return_value=payload):
        assert benchmark.main(["--sizes", "100", "--samples", "1"]) == 0

    output = capsys.readouterr().out
    assert json.loads(output) == payload["comparison"]


def test_physical_plaintext_size_uses_distinct_message_rows():
    metrics = benchmark.message_reuse_metrics(
        linked=(5, 100, 150, 2),
        physical=(40, 60),
    )

    assert metrics["physical_plaintext_size_sum"] == 40
    assert metrics["linked_logical_plaintext_bytes"] == 100
    assert metrics["reuse_ratio"] == 2.5
    assert metrics["physical_to_linked_ratio"] == 0.4


def test_a12_capacity_gate_checks_bytes_reuse_and_timing_hangs():
    message_bytes = benchmark.MIB
    case = {
        "workload": {
            "protocol": "S1",
            "completed_turns": 2,
            "target_history_bytes_per_turn": message_bytes,
            "logical_history_bytes_per_turn": message_bytes,
            "logical_history_bytes_total": 2 * message_bytes,
            "message_plaintext_bytes_each": message_bytes,
        },
        "timings": {
            "warm_cached_read": _timing(1.0),
            "warm_uncached_read": _timing(1.0),
            "warm_fold": _timing(1.0),
            "warm_append": _timing(1.0),
            "warm_drive": _timing(1.0),
        },
        "bytes_and_reuse": {
            "relations": {
                "sk_record": {"total_bytes": 1},
                "sk_content": {"total_bytes": 1},
                "sk_content_link": {"total_bytes": 1},
            },
            "message_reuse": {
                "unique_content_rows": 1,
                "link_rows": 2,
                "linked_logical_plaintext_bytes": 2 * message_bytes,
                "physical_plaintext_size_sum": message_bytes,
                "reuse_ratio": 2,
            },
        },
        "fixture_security": {
            "all_inputs_signed": True,
            "all_content_encrypted": True,
        },
        "cold_samples": 0,
    }
    payload = {
        "cold_mode": "none",
        "history": [],
        "runnable": [],
        "cleanup": {"database_created": 1, "database_dropped": 1},
        "a12": case,
    }

    checks = benchmark.capacity_failures(payload, a12_requested=True)
    assert all(checks.values())

    case["timings"]["warm_append"] = _timing(benchmark._A12_WARM_TIMING_LIMIT_MS + 1)
    checks = benchmark.capacity_failures(payload, a12_requested=True)
    assert not checks["a12:timing:warm_append"]
