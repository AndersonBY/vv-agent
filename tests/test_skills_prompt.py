from __future__ import annotations

from pathlib import Path

from v_agent.skills.prompt import metadata_to_prompt_entries, to_available_skills_xml


def test_to_available_skills_xml_includes_location_and_escaping(tmp_path: Path) -> None:
    skill_dir = tmp_path / "special-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: special-skill
description: Use <foo> & <bar> tags
---
Body
""",
        encoding="utf-8",
    )

    xml = to_available_skills_xml([skill_dir])
    assert "<available_skills>" in xml
    assert "<name>\nspecial-skill\n</name>" in xml
    assert "&lt;foo&gt;" in xml
    assert "&amp;" in xml
    assert "<location>" in xml


def test_metadata_to_prompt_entries_can_load_from_location(tmp_path: Path) -> None:
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

    entries = metadata_to_prompt_entries(
        [{"location": str(skill_dir)}],
        workspace=tmp_path,
    )
    assert len(entries) == 1
    assert entries[0].name == "my-skill"
    assert entries[0].description == "A test skill"
    assert entries[0].location is not None
    assert entries[0].location.endswith("/SKILL.md")


def test_metadata_to_prompt_entries_skips_invalid_entries(tmp_path: Path) -> None:
    entries = metadata_to_prompt_entries(
        [{"name": "", "description": ""}, {"location": "missing"}, ""],
        workspace=tmp_path,
    )
    assert entries == []
