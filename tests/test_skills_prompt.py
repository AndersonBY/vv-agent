from __future__ import annotations

from pathlib import Path

from vv_agent.skills.normalize import normalize_skill_list
from vv_agent.skills.prompt import render_skills_xml, skill_entry_to_xml, to_available_skills_xml


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


def test_normalize_skill_list_can_load_from_location(tmp_path: Path) -> None:
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

    entries = normalize_skill_list(
        [{"name": "my-skill", "description": "A test skill", "location": str(skill_dir)}],
        workspace=tmp_path,
    )
    assert len(entries) == 1
    assert entries[0].name == "my-skill"
    assert entries[0].description == "A test skill"


def test_normalize_skill_list_skips_invalid_entries(tmp_path: Path) -> None:
    entries = normalize_skill_list(
        [{"name": "", "description": ""}, {"location": "missing"}, ""],
        workspace=tmp_path,
    )
    assert entries == []


def test_normalize_skill_list_supports_skill_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / "alpha").mkdir(parents=True)
    (root / "beta").mkdir(parents=True)
    (root / "alpha" / "SKILL.md").write_text(
        """---
name: alpha
description: skill alpha
---
Body
""",
        encoding="utf-8",
    )
    (root / "beta" / "SKILL.md").write_text(
        """---
name: beta
description: skill beta
---
Body
""",
        encoding="utf-8",
    )

    entries = normalize_skill_list(["skills"], workspace=tmp_path)
    names = {item.name for item in entries}
    assert names == {"alpha", "beta"}


def test_render_skills_xml_truncates_on_budget() -> None:
    from vv_agent.skills.normalize import SkillEntry

    entries = [
        SkillEntry(name=f"skill-{i}", description=f"Description for skill {i}")
        for i in range(50)
    ]
    xml = render_skills_xml(entries, budget=500)
    assert "<available_skills>" in xml
    assert "more skills available" in xml
    assert xml.count("<skill>") < 50


def test_skill_entry_to_xml_omits_location_when_flag_false() -> None:
    from vv_agent.skills.normalize import SkillEntry

    entry = SkillEntry(name="test", description="desc", location="/path/to/SKILL.md")
    xml_with = skill_entry_to_xml(entry, include_location=True)
    xml_without = skill_entry_to_xml(entry, include_location=False)
    assert "<location>" in xml_with
    assert "<location>" not in xml_without
