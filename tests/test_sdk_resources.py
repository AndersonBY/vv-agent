from __future__ import annotations

from pathlib import Path

from v_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from v_agent.llm import ScriptedLLM
from v_agent.sdk import AgentDefinition, AgentResourceLoader, AgentSDKClient, AgentSDKOptions
from v_agent.tools import build_default_registry
from v_agent.types import AgentStatus, LLMResponse


def _fake_resolved(*, backend: str, model: str) -> ResolvedModelConfig:
    endpoint = EndpointConfig(endpoint_id="fake", api_key="k", api_base="https://example.invalid/v1")
    option = EndpointOption(endpoint=endpoint, model_id=model)
    return ResolvedModelConfig(
        backend=backend,
        requested_model=model,
        selected_model=model,
        model_id=model,
        endpoint_options=[option],
    )


def test_resource_loader_discovers_agents_prompts_skills_and_hooks(tmp_path: Path) -> None:
    resource_root = tmp_path / ".v-agent"
    (resource_root / "prompts").mkdir(parents=True)
    (resource_root / "skills" / "demo").mkdir(parents=True)
    (resource_root / "hooks").mkdir(parents=True)

    (resource_root / "agents.json").write_text(
        """{
  "profiles": {
    "researcher": {
      "description": "research profile",
      "model": "kimi-k2.5",
      "backend": "moonshot",
      "system_prompt_template": "research"
    }
  }
}
""",
        encoding="utf-8",
    )
    (resource_root / "prompts" / "research.md").write_text("You are loaded from template.", encoding="utf-8")
    (resource_root / "skills" / "demo" / "SKILL.md").write_text(
        """---
name: demo
description: demo skill
---
body
""",
        encoding="utf-8",
    )
    (resource_root / "hooks" / "noop.py").write_text(
        """from v_agent.runtime import BaseRuntimeHook, BeforeLLMEvent

class NoopHook(BaseRuntimeHook):
    def before_llm(self, event: BeforeLLMEvent):
        del event
        return None

HOOK = NoopHook()
""",
        encoding="utf-8",
    )

    loader = AgentResourceLoader(workspace=tmp_path, project_resource_dir=resource_root, global_resource_dir=tmp_path / ".none")
    discovered = loader.discover()

    assert "researcher" in discovered.agents
    assert discovered.prompts["research"] == "You are loaded from template."
    assert any(path.endswith("/skills") for path in discovered.skill_directories)
    assert len(discovered.hooks) == 1
    assert discovered.diagnostics == []


def test_sdk_client_uses_discovered_agent_and_prompt_template(tmp_path: Path) -> None:
    resource_root = tmp_path / ".v-agent"
    (resource_root / "prompts").mkdir(parents=True)
    (resource_root / "skills" / "demo").mkdir(parents=True)

    (resource_root / "agents.json").write_text(
        """{
  "profiles": {
    "researcher": {
      "description": "fallback",
      "model": "kimi-k2.5",
      "system_prompt_template": "research"
    }
  }
}
""",
        encoding="utf-8",
    )
    (resource_root / "prompts" / "research.md").write_text("Template system prompt", encoding="utf-8")
    (resource_root / "skills" / "demo" / "SKILL.md").write_text(
        """---
name: demo
description: demo skill
---
body
""",
        encoding="utf-8",
    )

    loader = AgentResourceLoader(workspace=tmp_path, project_resource_dir=resource_root, global_resource_dir=tmp_path / ".none")
    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            resource_loader=loader,
        )
    )

    task = client.prepare_task(
        agent="researcher",
        prompt="hi",
        resolved_model_id="kimi-k2.5",
    )
    assert "Template system prompt" in task.system_prompt
    assert "available_skills" in task.metadata
    assert "<name>\ndemo\n</name>" in task.system_prompt


def test_sdk_client_loads_runtime_hooks_from_resource_loader(tmp_path: Path) -> None:
    resource_root = tmp_path / ".v-agent"
    (resource_root / "hooks").mkdir(parents=True)
    (resource_root / "hooks" / "force_finish.py").write_text(
        """from v_agent.constants import TASK_FINISH_TOOL_NAME
from v_agent.runtime import AfterLLMEvent, BaseRuntimeHook
from v_agent.types import LLMResponse, ToolCall

class ForceFinishHook(BaseRuntimeHook):
    def after_llm(self, event: AfterLLMEvent):
        return LLMResponse(
            content=event.response.content,
            tool_calls=[ToolCall(id="h1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "hook-finish"})],
        )

HOOK = ForceFinishHook()
""",
        encoding="utf-8",
    )
    loader = AgentResourceLoader(workspace=tmp_path, project_resource_dir=resource_root, global_resource_dir=tmp_path / ".none")

    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        return ScriptedLLM(steps=[LLMResponse(content="plain")]), _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
            resource_loader=loader,
        ),
        agent=AgentDefinition(description="helper", model="kimi-k2.5"),
    )

    run = client.run(prompt="hello")
    assert run.result.status == AgentStatus.COMPLETED
    assert run.result.final_answer == "hook-finish"
