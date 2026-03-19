from __future__ import annotations

from pathlib import Path

from vv_agent.skills.normalize import normalize_skill_list


def _write_skill(skill_dir: Path, name: str, description: str, body: str = "Body") -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


def test_normalize_str_path(tmp_path: Path) -> None:
    _write_skill(tmp_path / "my-skill", "my-skill", "A test skill")
    entries = normalize_skill_list([str(tmp_path / "my-skill")])
    assert len(entries) == 1
    assert entries[0].name == "my-skill"
    assert entries[0].description == "A test skill"
    assert entries[0].instructions is None


def test_normalize_str_path_with_instructions(tmp_path: Path) -> None:
    _write_skill(tmp_path / "my-skill", "my-skill", "A test skill", body="Do this thing")
    entries = normalize_skill_list([str(tmp_path / "my-skill")], load_instructions=True)
    assert len(entries) == 1
    assert entries[0].instructions == "Do this thing"


def test_normalize_str_relative_to_workspace(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills" / "demo", "demo", "Demo skill")
    entries = normalize_skill_list(["skills/demo"], workspace=tmp_path)
    assert len(entries) == 1
    assert entries[0].name == "demo"


def test_normalize_str_discovers_children(tmp_path: Path) -> None:
    root = tmp_path / "all-skills"
    _write_skill(root / "alpha", "alpha", "Alpha")
    _write_skill(root / "beta", "beta", "Beta")
    entries = normalize_skill_list([str(root)])
    names = {e.name for e in entries}
    assert names == {"alpha", "beta"}


def test_normalize_dict_with_name_description() -> None:
    entries = normalize_skill_list([{"name": "foo", "description": "Foo skill"}])
    assert len(entries) == 1
    assert entries[0].name == "foo"
    assert entries[0].description == "Foo skill"


def test_normalize_dict_with_location_fallback(tmp_path: Path) -> None:
    _write_skill(tmp_path / "bar", "bar", "Bar skill")
    entries = normalize_skill_list(
        [{"location": str(tmp_path / "bar")}],
        workspace=tmp_path,
    )
    assert len(entries) == 1
    assert entries[0].name == "bar"


def test_normalize_deduplicates_by_name(tmp_path: Path) -> None:
    _write_skill(tmp_path / "dup", "dup", "First")
    entries = normalize_skill_list([
        str(tmp_path / "dup"),
        {"name": "dup", "description": "Second"},
    ])
    assert len(entries) == 1
    assert entries[0].description == "First"


def test_normalize_skips_missing_path() -> None:
    entries = normalize_skill_list(["/nonexistent/path"])
    assert entries == []


def test_normalize_skips_empty_dict() -> None:
    entries = normalize_skill_list([{"name": "", "description": ""}])
    assert entries == []


def test_normalize_returns_empty_for_none() -> None:
    assert normalize_skill_list(None) == []


def test_normalize_dict_with_instructions() -> None:
    entries = normalize_skill_list([{
        "name": "manual",
        "description": "Manual skill",
        "instructions": "Do the thing",
        "compatibility": "python>=3.10",
        "allowed-tools": "bash read_file",
        "metadata": {"author": "test"},
    }])
    assert len(entries) == 1
    e = entries[0]
    assert e.name == "manual"
    assert e.instructions == "Do the thing"
    assert e.compatibility == "python>=3.10"
    assert e.allowed_tools == "bash read_file"
    assert e.metadata == {"author": "test"}


def test_normalize_location_relative_to_workspace(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills" / "rel", "rel", "Relative")
    entries = normalize_skill_list(
        [str(tmp_path / "skills" / "rel")],
        workspace=tmp_path,
    )
    assert entries[0].location is not None
    assert entries[0].location.startswith("skills/")
