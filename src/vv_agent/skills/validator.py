from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vv_agent.skills.errors import SkillParseError

MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500

ALLOWED_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "allowed-tools",
    "metadata",
}


ValidationMode = Literal["strict", "compat", "minimal"]
IssueSeverity = Literal["error", "warning", "ignore"]

DEFAULT_VALIDATION_MODE: ValidationMode = "strict"
VALIDATION_MODES: tuple[ValidationMode, ...] = ("strict", "compat", "minimal")


@dataclass(slots=True)
class ValidationDiagnostics:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalize_validation_mode(validation_mode: str | None) -> ValidationMode:
    normalized = str(validation_mode or DEFAULT_VALIDATION_MODE).strip().lower()
    if normalized == "strict":
        return "strict"
    if normalized == "compat":
        return "compat"
    if normalized == "minimal":
        return "minimal"
    raise ValueError(
        f"Unsupported validation mode '{validation_mode}'. Expected one of {list(VALIDATION_MODES)}."
    )


def _append_issue(diagnostics: ValidationDiagnostics, message: str, *, severity: IssueSeverity) -> None:
    if severity == "ignore":
        return
    if severity == "warning":
        diagnostics.warnings.append(message)
        return
    diagnostics.errors.append(message)


def _merge_diagnostics(base: ValidationDiagnostics, incoming: ValidationDiagnostics) -> None:
    base.errors.extend(incoming.errors)
    base.warnings.extend(incoming.warnings)


def _validate_name(
    name: Any,
    *,
    validation_mode: ValidationMode,
    skill_dir: Path | None = None,
) -> ValidationDiagnostics:
    diagnostics = ValidationDiagnostics()
    if not isinstance(name, str) or not name.strip():
        diagnostics.errors.append("Field 'name' must be a non-empty string")
        return diagnostics

    normalized = unicodedata.normalize("NFKC", name.strip())
    if len(normalized) > MAX_SKILL_NAME_LENGTH:
        diagnostics.errors.append(
            f"Skill name '{normalized}' exceeds {MAX_SKILL_NAME_LENGTH} character limit ({len(normalized)} chars)"
        )

    if not all(ch.isalnum() or ch == "-" for ch in normalized):
        diagnostics.errors.append(
            f"Skill name '{normalized}' contains invalid characters. Only letters, digits, and hyphens are allowed."
        )

    naming_severity: IssueSeverity = "error" if validation_mode in {"strict", "compat"} else "warning"
    if normalized != normalized.lower():
        _append_issue(
            diagnostics,
            f"Skill name '{normalized}' must be lowercase",
            severity=naming_severity,
        )
    if normalized.startswith("-") or normalized.endswith("-"):
        _append_issue(
            diagnostics,
            "Skill name cannot start or end with a hyphen",
            severity=naming_severity,
        )
    if "--" in normalized:
        _append_issue(
            diagnostics,
            "Skill name cannot contain consecutive hyphens",
            severity=naming_severity,
        )

    if skill_dir is not None:
        dir_name = unicodedata.normalize("NFKC", skill_dir.name)
        if dir_name != normalized:
            _append_issue(
                diagnostics,
                f"Directory name '{skill_dir.name}' must match skill name '{normalized}'",
                severity="error" if validation_mode == "strict" else "warning",
            )

    return diagnostics


def _validate_description(description: Any) -> list[str]:
    if not isinstance(description, str) or not description.strip():
        return ["Field 'description' must be a non-empty string"]
    if len(description) > MAX_DESCRIPTION_LENGTH:
        return [f"Description exceeds {MAX_DESCRIPTION_LENGTH} character limit ({len(description)} chars)"]
    return []


def _validate_compatibility(compatibility: Any) -> list[str]:
    if compatibility is None:
        return []
    if not isinstance(compatibility, str):
        return ["Field 'compatibility' must be a string"]
    if len(compatibility) > MAX_COMPATIBILITY_LENGTH:
        return [f"Compatibility exceeds {MAX_COMPATIBILITY_LENGTH} character limit ({len(compatibility)} chars)"]
    return []


def validate_metadata_with_diagnostics(
    metadata: dict[str, Any],
    *,
    skill_dir: Path | None = None,
    validation_mode: str | None = DEFAULT_VALIDATION_MODE,
) -> ValidationDiagnostics:
    mode = normalize_validation_mode(validation_mode)
    diagnostics = ValidationDiagnostics()

    extra_fields = set(metadata.keys()) - ALLOWED_FIELDS
    if extra_fields:
        _append_issue(
            diagnostics,
            f"Unexpected fields in frontmatter: {', '.join(sorted(extra_fields))}. "
            f"Only {sorted(ALLOWED_FIELDS)} are allowed.",
            severity="error" if mode == "strict" else "warning",
        )

    if "name" not in metadata:
        diagnostics.errors.append("Missing required field in frontmatter: name")
    else:
        _merge_diagnostics(
            diagnostics,
            _validate_name(
                metadata["name"],
                validation_mode=mode,
                skill_dir=skill_dir,
            ),
        )

    if "description" not in metadata:
        diagnostics.errors.append("Missing required field in frontmatter: description")
    else:
        diagnostics.errors.extend(_validate_description(metadata["description"]))

    compatibility_errors = _validate_compatibility(metadata.get("compatibility"))
    compatibility_severity: IssueSeverity = "error" if mode in {"strict", "compat"} else "warning"
    for message in compatibility_errors:
        _append_issue(diagnostics, message, severity=compatibility_severity)
    return diagnostics


def validate_metadata(
    metadata: dict[str, Any],
    *,
    skill_dir: Path | None = None,
    validation_mode: str | None = DEFAULT_VALIDATION_MODE,
) -> list[str]:
    return validate_metadata_with_diagnostics(
        metadata,
        skill_dir=skill_dir,
        validation_mode=validation_mode,
    ).errors


def validate_with_diagnostics(
    skill_dir: Path,
    *,
    validation_mode: str | None = DEFAULT_VALIDATION_MODE,
) -> ValidationDiagnostics:
    """Validate a skill directory path and return diagnostics."""
    diagnostics = ValidationDiagnostics()
    mode = normalize_validation_mode(validation_mode)
    skill_dir = Path(skill_dir)

    if not skill_dir.exists():
        diagnostics.errors.append(f"Path does not exist: {skill_dir}")
        return diagnostics
    if not skill_dir.is_dir():
        diagnostics.errors.append(f"Not a directory: {skill_dir}")
        return diagnostics

    from vv_agent.skills.parser import find_skill_md, parse_frontmatter

    skill_md = find_skill_md(skill_dir)
    if skill_md is None:
        diagnostics.errors.append("Missing required file: SKILL.md")
        return diagnostics

    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
        metadata, _ = parse_frontmatter(content)
    except SkillParseError as exc:
        diagnostics.errors.append(str(exc))
        return diagnostics

    return validate_metadata_with_diagnostics(
        metadata,
        skill_dir=skill_dir,
        validation_mode=mode,
    )


def validate(
    skill_dir: Path,
    *,
    validation_mode: str | None = DEFAULT_VALIDATION_MODE,
) -> list[str]:
    """Validate a skill directory path and return all validation errors."""
    return validate_with_diagnostics(skill_dir, validation_mode=validation_mode).errors
