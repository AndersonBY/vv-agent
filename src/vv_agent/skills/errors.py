from __future__ import annotations


class SkillError(Exception):
    """Base error for skill parsing/validation."""


class SkillParseError(SkillError):
    """Raised when SKILL.md frontmatter is malformed."""


class SkillValidationError(SkillError):
    """Raised when skill metadata violates specification constraints."""
