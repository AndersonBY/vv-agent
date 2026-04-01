from __future__ import annotations

from pathlib import Path

from vv_agent.memory.post_compact_restore import PostCompactRestoreConfig, restore_key_files


def test_restore_key_files_prioritizes_modified_then_created(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('b')\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("print('c')\n", encoding="utf-8")

    restored = restore_key_files(
        {
            "files_examined_or_modified": [
                {"path": "c.py", "action": "read", "summary": "read c"},
                {"path": "a.py", "action": "modified", "summary": "updated a"},
                {"path": "b.py", "action": "created", "summary": "created b"},
            ]
        },
        tmp_path,
        PostCompactRestoreConfig(max_files=2),
    )

    assert restored.index('path="a.py"') < restored.index('path="b.py"')
    assert 'path="c.py"' not in restored


def test_restore_key_files_respects_single_file_budget(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("x = 1\n" * 400, encoding="utf-8")

    restored = restore_key_files(
        {
            "files_examined_or_modified": [
                {"path": "big.py", "action": "modified", "summary": "updated big"},
            ]
        },
        tmp_path,
        PostCompactRestoreConfig(max_tokens_per_file=40, total_budget_tokens=200),
    )

    assert "<Post-Compaction File Context>" in restored
    assert "truncated after compaction restore" in restored


def test_restore_key_files_respects_total_budget(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\n" * 200, encoding="utf-8")
    (tmp_path / "b.py").write_text("b = 2\n" * 200, encoding="utf-8")

    restored = restore_key_files(
        {
            "files_examined_or_modified": [
                {"path": "a.py", "action": "modified", "summary": "updated a"},
                {"path": "b.py", "action": "created", "summary": "created b"},
            ]
        },
        tmp_path,
        PostCompactRestoreConfig(total_budget_tokens=180, max_tokens_per_file=120),
    )

    assert 'path="a.py"' in restored
    assert 'path="b.py"' not in restored


def test_restore_key_files_skips_missing_and_escaped_paths(tmp_path: Path) -> None:
    (tmp_path / "safe.py").write_text("print('safe')\n", encoding="utf-8")

    restored = restore_key_files(
        {
            "files_examined_or_modified": [
                {"path": "../../etc/passwd", "action": "modified", "summary": "bad"},
                {"path": "missing.py", "action": "read", "summary": "missing"},
                {"path": "safe.py", "action": "read", "summary": "safe"},
            ]
        },
        tmp_path,
    )

    assert "../../etc/passwd" not in restored
    assert "missing.py" not in restored
    assert 'path="safe.py"' in restored
