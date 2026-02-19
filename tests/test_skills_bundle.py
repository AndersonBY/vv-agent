from __future__ import annotations

from pathlib import Path

import pytest

from v_agent.skills.bundle import PreparedSkill, prepare_skill_bundle


def _make_skill(root: Path, name: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}\n---\nBody\n",
        encoding="utf-8",
    )
    return skill_dir


def test_prepare_skill_bundle_basic(tmp_path: Path) -> None:
    source = tmp_path / "skills"
    source.mkdir()
    _make_skill(source, "alpha")
    _make_skill(source, "beta")

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = prepare_skill_bundle(source, workspace)
    assert len(result) == 2
    assert all(isinstance(s, PreparedSkill) for s in result)
    names = {s.name for s in result}
    assert names == {"alpha", "beta"}
    for s in result:
        assert s.runtime.exists()
        assert (s.runtime / "SKILL.md").exists()


def test_prepare_skill_bundle_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "skills"
    source.mkdir()
    _make_skill(source, "dup")
    nested = source / "sub"
    nested.mkdir()
    _make_skill(nested, "dup")

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = prepare_skill_bundle(source, workspace)
    assert len(result) == 1
    assert result[0].name == "dup"


def test_prepare_skill_bundle_missing_source(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(FileNotFoundError):
        prepare_skill_bundle(tmp_path / "nonexistent", workspace)


def test_prepare_skill_bundle_no_skills(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="No SKILL.md"):
        prepare_skill_bundle(source, workspace)
