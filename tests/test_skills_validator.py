from __future__ import annotations

from pathlib import Path

from v_agent.skills.validator import validate, validate_metadata


def test_validate_skill_dir_success(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: my-skill
description: A test skill
---
Body
""",
        encoding="utf-8",
    )

    assert validate(skill_dir) == []


def test_validate_skill_dir_reports_missing_file(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    errors = validate(skill_dir)
    assert errors == ["Missing required file: SKILL.md"]


def test_validate_metadata_rejects_unknown_fields() -> None:
    errors = validate_metadata(
        {
            "name": "my-skill",
            "description": "A test skill",
            "unknown": "value",
        }
    )
    assert any("Unexpected fields" in message for message in errors)


def test_validate_metadata_rejects_uppercase_name() -> None:
    errors = validate_metadata(
        {
            "name": "My-Skill",
            "description": "A test skill",
        }
    )
    assert any("must be lowercase" in message for message in errors)


def test_validate_metadata_accepts_i18n_lowercase() -> None:
    skill_dir = Path("мой-навык")
    errors = validate_metadata(
        {
            "name": "мой-навык",
            "description": "A test skill",
        },
        skill_dir=skill_dir,
    )
    assert errors == []
