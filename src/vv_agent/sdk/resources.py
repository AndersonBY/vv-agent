from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vv_agent.runtime.hooks import RuntimeHook
from vv_agent.sdk.types import AgentDefinition
from vv_agent.types import NoToolPolicy, SubAgentConfig

_HOOK_METHODS = (
    "before_memory_compact",
    "before_llm",
    "after_llm",
    "before_tool_call",
    "after_tool_call",
)


@dataclass(slots=True)
class DiscoveredResources:
    agents: dict[str, AgentDefinition] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    skill_directories: list[str] = field(default_factory=list)
    hooks: list[RuntimeHook] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


class AgentResourceLoader:
    """Auto-discover agent resources from global and project directories."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        project_resource_dir: str | Path | None = None,
        global_resource_dir: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.project_resource_dir = (
            Path(project_resource_dir).resolve()
            if project_resource_dir is not None
            else (self.workspace / ".vv-agent").resolve()
        )
        self.global_resource_dir = (
            Path(global_resource_dir).expanduser().resolve()
            if global_resource_dir is not None
            else Path("~/.vv-agent").expanduser().resolve()
        )
        self._cached: DiscoveredResources | None = None

    def discover(self, *, force_reload: bool = False) -> DiscoveredResources:
        if self._cached is not None and not force_reload:
            return self._cached

        discovered = DiscoveredResources()
        for root in self._resource_roots():
            self._load_agents(root=root, target=discovered)
            self._load_prompts(root=root, target=discovered)
            self._load_skills(root=root, target=discovered)
            self._load_hooks(root=root, target=discovered)

        self._cached = discovered
        return discovered

    def _resource_roots(self) -> list[Path]:
        roots: list[Path] = []
        for root in (self.global_resource_dir, self.project_resource_dir):
            if root.exists() and root.is_dir():
                roots.append(root)
        return roots

    def _load_agents(self, *, root: Path, target: DiscoveredResources) -> None:
        config_file = root / "agents.json"
        if not config_file.exists() or not config_file.is_file():
            return

        try:
            raw = json.loads(config_file.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            target.diagnostics.append(f"Invalid agents.json in {root}: {exc}")
            return

        profiles = raw.get("profiles") if isinstance(raw, dict) and isinstance(raw.get("profiles"), dict) else raw
        if not isinstance(profiles, dict):
            target.diagnostics.append(f"agents.json in {root} must be an object or contain `profiles` object.")
            return

        for name, payload in profiles.items():
            if not isinstance(name, str) or not name.strip():
                target.diagnostics.append(f"Skip invalid profile name in {config_file}.")
                continue
            if not isinstance(payload, dict):
                target.diagnostics.append(f"Skip profile `{name}` in {config_file}: definition must be an object.")
                continue
            definition = self._parse_agent_definition(
                profile_name=name,
                payload=payload,
                base_dir=config_file.parent,
                diagnostics=target.diagnostics,
            )
            if definition is not None:
                target.agents[name] = definition

    def _parse_agent_definition(
        self,
        *,
        profile_name: str,
        payload: dict[str, Any],
        base_dir: Path,
        diagnostics: list[str],
    ) -> AgentDefinition | None:
        description = payload.get("description")
        model = payload.get("model")
        if not isinstance(description, str) or not description.strip():
            diagnostics.append(f"Skip profile `{profile_name}`: `description` must be non-empty string.")
            return None
        if not isinstance(model, str) or not model.strip():
            diagnostics.append(f"Skip profile `{profile_name}`: `model` must be non-empty string.")
            return None

        sub_agents: dict[str, SubAgentConfig] = {}
        raw_sub_agents = payload.get("sub_agents")
        if isinstance(raw_sub_agents, dict):
            for sub_name, sub_payload in raw_sub_agents.items():
                if not isinstance(sub_name, str) or not isinstance(sub_payload, dict):
                    continue
                sub_model = sub_payload.get("model")
                sub_description = sub_payload.get("description")
                if not isinstance(sub_model, str) or not isinstance(sub_description, str):
                    continue
                sub_agents[sub_name] = SubAgentConfig(
                    model=sub_model,
                    description=sub_description,
                    backend=sub_payload.get("backend") if isinstance(sub_payload.get("backend"), str) else None,
                    system_prompt=sub_payload.get("system_prompt")
                    if isinstance(sub_payload.get("system_prompt"), str)
                    else None,
                    max_cycles=int(sub_payload.get("max_cycles", 8)) if isinstance(sub_payload.get("max_cycles", 8), int) else 8,
                    exclude_tools=list(sub_payload.get("exclude_tools", []))
                    if isinstance(sub_payload.get("exclude_tools"), list)
                    else [],
                    metadata=dict(sub_payload.get("metadata", {})) if isinstance(sub_payload.get("metadata"), dict) else {},
                )

        skill_directories: list[str] = []
        raw_skill_directories = payload.get("skill_directories")
        if isinstance(raw_skill_directories, list):
            for item in raw_skill_directories:
                if not isinstance(item, str) or not item.strip():
                    continue
                skill_directories.append(self._resolve_path(base_dir, item.strip()))

        language_raw = payload.get("language")
        language = language_raw if isinstance(language_raw, str) and language_raw.strip() else "zh-CN"
        raw_no_tool_policy = payload.get("no_tool_policy")
        no_tool_policy: NoToolPolicy = "continue"
        if raw_no_tool_policy in {"continue", "wait_user", "finish"}:
            no_tool_policy = raw_no_tool_policy

        return AgentDefinition(
            description=description.strip(),
            model=model.strip(),
            backend=payload.get("backend") if isinstance(payload.get("backend"), str) else None,
            language=language,
            max_cycles=int(payload.get("max_cycles", 10)) if isinstance(payload.get("max_cycles", 10), int) else 10,
            no_tool_policy=no_tool_policy,
            allow_interruption=bool(payload.get("allow_interruption", True)),
            use_workspace=bool(payload.get("use_workspace", True)),
            enable_todo_management=bool(payload.get("enable_todo_management", True)),
            agent_type=payload.get("agent_type") if isinstance(payload.get("agent_type"), str) else None,
            native_multimodal=bool(payload.get("native_multimodal", False)),
            enable_sub_agents=bool(payload.get("enable_sub_agents", False)),
            sub_agents=sub_agents,
            skill_directories=skill_directories,
            extra_tool_names=(
                list(payload.get("extra_tool_names", []))
                if isinstance(payload.get("extra_tool_names"), list)
                else []
            ),
            exclude_tools=list(payload.get("exclude_tools", [])) if isinstance(payload.get("exclude_tools"), list) else [],
            bash_shell=payload.get("bash_shell") if isinstance(payload.get("bash_shell"), str) else None,
            windows_shell_priority=(
                [str(item).strip() for item in payload.get("windows_shell_priority", []) if str(item).strip()]
                if isinstance(payload.get("windows_shell_priority"), list)
                else []
            ),
            bash_env=(
                {
                    str(key).strip(): str(value)
                    for key, value in payload.get("bash_env", {}).items()
                    if str(key).strip()
                }
                if isinstance(payload.get("bash_env"), dict)
                else {}
            ),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
            system_prompt=payload.get("system_prompt") if isinstance(payload.get("system_prompt"), str) else None,
            system_prompt_template=payload.get("system_prompt_template")
            if isinstance(payload.get("system_prompt_template"), str)
            else None,
        )

    def _load_prompts(self, *, root: Path, target: DiscoveredResources) -> None:
        prompts_dir = root / "prompts"
        if not prompts_dir.exists() or not prompts_dir.is_dir():
            return
        for prompt_file in sorted(prompts_dir.glob("*.md")):
            target.prompts[prompt_file.stem] = prompt_file.read_text(encoding="utf-8", errors="replace")

    def _load_skills(self, *, root: Path, target: DiscoveredResources) -> None:
        skills_dir = root / "skills"
        if not skills_dir.exists() or not skills_dir.is_dir():
            return
        resolved = skills_dir.resolve().as_posix()
        if resolved not in target.skill_directories:
            target.skill_directories.append(resolved)

    def _load_hooks(self, *, root: Path, target: DiscoveredResources) -> None:
        hooks_dir = root / "hooks"
        if not hooks_dir.exists() or not hooks_dir.is_dir():
            return

        hook_files: list[Path] = sorted(path for path in hooks_dir.glob("*.py") if path.is_file())
        hook_files.extend(
            sorted(path / "index.py" for path in hooks_dir.glob("*") if path.is_dir() and (path / "index.py").is_file())
        )

        for hook_file in hook_files:
            for hook in self._load_hook_objects(hook_file, diagnostics=target.diagnostics):
                target.hooks.append(hook)

    def _load_hook_objects(self, hook_file: Path, *, diagnostics: list[str]) -> list[RuntimeHook]:
        module_name = f"vv_agent_user_hook_{abs(hash(hook_file.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, hook_file)
        if spec is None or spec.loader is None:
            diagnostics.append(f"Cannot load hook module: {hook_file}")
            return []

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            diagnostics.append(f"Hook load failed ({hook_file}): {exc}")
            return []

        candidates: list[Any] = []
        if hasattr(module, "create_hook") and callable(module.create_hook):
            try:
                created = module.create_hook()
            except Exception as exc:
                diagnostics.append(f"Hook factory failed ({hook_file}): {exc}")
                created = None
            if isinstance(created, list):
                candidates.extend(created)
            elif created is not None:
                candidates.append(created)

        for attr_name in ("HOOK", "HOOKS"):
            if not hasattr(module, attr_name):
                continue
            value = getattr(module, attr_name)
            if isinstance(value, list):
                candidates.extend(value)
            else:
                candidates.append(value)

        hooks: list[RuntimeHook] = []
        for candidate in candidates:
            if self._looks_like_hook(candidate):
                hooks.append(candidate)
            else:
                diagnostics.append(f"Skip invalid hook object from {hook_file}: {type(candidate).__name__}")
        return hooks

    @staticmethod
    def _looks_like_hook(candidate: Any) -> bool:
        return any(callable(getattr(candidate, name, None)) for name in _HOOK_METHODS)

    @staticmethod
    def _resolve_path(base_dir: Path, raw_path: str) -> str:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path.as_posix()
