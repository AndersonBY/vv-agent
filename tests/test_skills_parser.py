from __future__ import annotations

from pathlib import Path

import pytest

from v_agent.skills import SkillParseError, SkillValidationError
from v_agent.skills.parser import find_skill_md, parse_frontmatter, read_properties, read_skill


def test_parse_frontmatter_valid() -> None:
    content = """---
name: my-skill
description: A test skill
---
# My Skill
Instructions here.
"""
    metadata, body = parse_frontmatter(content)
    assert metadata["name"] == "my-skill"
    assert metadata["description"] == "A test skill"
    assert "# My Skill" in body


def test_parse_frontmatter_missing_block() -> None:
    with pytest.raises(SkillParseError, match="must start with YAML frontmatter"):
        parse_frontmatter("# No frontmatter")


def test_parse_frontmatter_unclosed() -> None:
    content = """---
name: my-skill
description: A test skill
"""
    with pytest.raises(SkillParseError, match="not properly closed"):
        parse_frontmatter(content)


def test_read_properties_loads_required_fields(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: my-skill
description: A test skill
allowed-tools: Bash(jq:*)
metadata:
  author: test
---
Body
""",
        encoding="utf-8",
    )

    props = read_properties(skill_dir)
    assert props.name == "my-skill"
    assert props.description == "A test skill"
    assert props.allowed_tools == "Bash(jq:*)"
    assert props.metadata == {"author": "test"}


def test_read_properties_requires_name_and_description(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
description: A test skill
---
Body
""",
        encoding="utf-8",
    )

    with pytest.raises(SkillValidationError, match="required field in frontmatter: name"):
        read_properties(skill_dir)


def test_find_skill_md_prefers_uppercase(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("uppercase", encoding="utf-8")
    (skill_dir / "skill.md").write_text("lowercase", encoding="utf-8")

    found = find_skill_md(skill_dir)
    assert found is not None
    assert found.name == "SKILL.md"


def test_read_skill_validates_directory_name_match(tmp_path: Path) -> None:
    skill_dir = tmp_path / "wrong-name"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: correct-name
description: A test skill
---
Body
""",
        encoding="utf-8",
    )

    with pytest.raises(SkillValidationError, match="must match skill name"):
        read_skill(skill_dir)
