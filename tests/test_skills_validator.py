from __future__ import annotations

from pathlib import Path

import pytest

from vv_agent.skills.validator import (
    normalize_validation_mode,
    validate,
    validate_metadata,
    validate_metadata_with_diagnostics,
)


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


def test_validate_metadata_compat_mode_demotes_extra_fields_and_dir_mismatch() -> None:
    diagnostics = validate_metadata_with_diagnostics(
        {
            "name": "tavily",
            "description": "A third-party skill package",
            "homepage": "https://example.com",
        },
        skill_dir=Path("5e4a40157abd"),
        validation_mode="compat",
    )
    assert diagnostics.errors == []
    assert any("Unexpected fields in frontmatter: homepage" in message for message in diagnostics.warnings)
    assert any("Directory name '5e4a40157abd' must match skill name 'tavily'" in message for message in diagnostics.warnings)


def test_validate_metadata_minimal_mode_demotes_naming_conventions() -> None:
    diagnostics = validate_metadata_with_diagnostics(
        {
            "name": "My-Skill",
            "description": "A test skill",
        },
        validation_mode="minimal",
    )
    assert diagnostics.errors == []
    assert any("must be lowercase" in message for message in diagnostics.warnings)


def test_normalize_validation_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported validation mode"):
        normalize_validation_mode("relaxed")
