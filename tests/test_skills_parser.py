from __future__ import annotations

from pathlib import Path

import pytest

from vv_agent.skills import SkillParseError, SkillValidationError
from vv_agent.skills.parser import find_skill_md, parse_frontmatter, read_properties, read_skill


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


@pytest.mark.parametrize(
    ("description_line", "expected"),
    [
        ("description: Compare A --- B", "Compare A --- B"),
        ('description: "Compare A --- B"', "Compare A --- B"),
    ],
)
def test_parse_frontmatter_preserves_delimiter_like_scalar_values(description_line: str, expected: str) -> None:
    content = f"---\nname: test-skill\n{description_line}\n---\nBody\n"

    metadata, body = parse_frontmatter(content)

    assert metadata["description"] == expected
    assert body == "Body"


def test_parse_frontmatter_preserves_delimiter_like_block_scalar_lines() -> None:
    content = """---
name: test-skill
description: |
  Compare A
  ---
  with B.
---
Body
"""

    metadata, body = parse_frontmatter(content)

    assert metadata["description"] == "Compare A\n---\nwith B.\n"
    assert body == "Body"


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_parse_frontmatter_accepts_marker_trailing_whitespace(newline: str) -> None:
    content = newline.join(
        [
            "---   ",
            "name: test-skill",
            "description: A test skill",
            "---\t ",
            "Body",
            "Second line",
        ]
    )

    metadata, body = parse_frontmatter(content)

    assert metadata["name"] == "test-skill"
    assert body == f"Body{newline}Second line"


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


def test_parse_frontmatter_rejects_closing_marker_with_body() -> None:
    content = """---
name: my-skill
description: A test skill
--- Body
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


def test_read_skill_compat_mode_allows_common_third_party_drift(tmp_path: Path) -> None:
    skill_dir = tmp_path / "5e4a40157abd"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: tavily
description: Tavily skill package
homepage: https://example.com
---
Body
""",
        encoding="utf-8",
    )

    loaded = read_skill(skill_dir, validation_mode="compat")
    assert loaded.properties.name == "tavily"
    assert any("Unexpected fields in frontmatter: homepage" in message for message in loaded.warnings)
    assert any("must match skill name 'tavily'" in message for message in loaded.warnings)


def test_discover_skill_dirs_from_root(tmp_path: Path) -> None:
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
    (root / "beta" / "skill.md").write_text(
        """---
name: beta
description: skill beta
---
Body
""",
        encoding="utf-8",
    )

    from vv_agent.skills.parser import discover_skill_dirs

    discovered = discover_skill_dirs(root)
    names = {path.name for path in discovered}
    assert names == {"alpha", "beta"}
