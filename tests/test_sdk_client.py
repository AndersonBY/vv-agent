from __future__ import annotations

from pathlib import Path

import pytest

from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.constants import (
    ASK_USER_TOOL_NAME,
    BATCH_SUB_TASKS_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
)
from vv_agent.llm import ScriptedLLM
from vv_agent.sdk import (
    AgentDefinition,
    AgentSDKClient,
    AgentSDKOptions,
)
from vv_agent.sdk import (
    query as sdk_query,
)
from vv_agent.sdk import (
    run as sdk_run,
)
from vv_agent.tools import build_default_registry
from vv_agent.types import AgentStatus, LLMResponse, SubAgentConfig, ToolCall


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


def test_sdk_client_runs_named_agent(tmp_path: Path) -> None:
    builder_calls: list[tuple[str, str, str]] = []

    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del timeout_seconds
        builder_calls.append((str(settings_path), backend, model))
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "ok"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agents={
            "demo": AgentDefinition(
                description="you are helper",
                model="kimi-k2.5",
            )
        },
    )

    run = client.run_agent(agent_name="demo", prompt="say ok")
    assert run.result.status == AgentStatus.COMPLETED
    assert run.result.final_answer == "ok"
    assert run.resolved.backend == "moonshot"
    assert builder_calls == [("local_settings.py", "moonshot", "kimi-k2.5")]


def test_sdk_client_runs_default_agent_without_name(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "default-ok"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agent=AgentDefinition(description="you are helper", model="kimi-k2.5"),
    )

    run = client.run(prompt="say ok")
    assert run.agent_name == "default"
    assert run.result.status == AgentStatus.COMPLETED
    assert run.result.final_answer == "default-ok"


def test_sdk_client_run_accepts_inline_agent_definition(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "inline-ok"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
    )

    run = client.run(
        prompt="say ok",
        agent=AgentDefinition(description="inline helper", model="kimi-k2.5"),
    )
    assert run.agent_name == "inline"
    assert run.result.status == AgentStatus.COMPLETED
    assert run.result.final_answer == "inline-ok"


def test_sdk_client_run_requires_agent_when_multiple_profiles(tmp_path: Path) -> None:
    client = AgentSDKClient(
        options=AgentSDKOptions(settings_file=Path("local_settings.py"), default_backend="moonshot", workspace=tmp_path),
        agents={
            "a": AgentDefinition(description="helper a", model="m"),
            "b": AgentDefinition(description="helper b", model="m"),
        },
    )
    with pytest.raises(ValueError, match="Multiple agents configured"):
        client.run(prompt="hi")


def test_sdk_client_run_auto_selects_only_profile(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "auto-profile"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agents={"only": AgentDefinition(description="helper", model="kimi-k2.5")},
    )

    run = client.run(prompt="say ok")
    assert run.agent_name == "only"
    assert run.result.final_answer == "auto-profile"


def test_sdk_client_rejects_conflicting_agent_selectors(tmp_path: Path) -> None:
    client = AgentSDKClient(
        options=AgentSDKOptions(settings_file=Path("local_settings.py"), default_backend="moonshot", workspace=tmp_path),
        agents={"demo": AgentDefinition(description="helper", model="m")},
    )
    with pytest.raises(ValueError, match="Use either 'agent' or 'agent_name'"):
        client.run(
            prompt="hi",
            agent="demo",
            agent_name="demo",
        )


def test_sdk_prepare_task_supports_sub_agent_configs(tmp_path: Path) -> None:
    client = AgentSDKClient(
        options=AgentSDKOptions(settings_file=Path("local_settings.py"), default_backend="moonshot", workspace=tmp_path),
        agents={
            "orchestrator": AgentDefinition(
                description="orchestrate",
                model="kimi-k2.5",
                sub_agents={
                    "research-sub": SubAgentConfig(
                        model="kimi-k2.5",
                        description="collect data",
                    )
                },
            )
        },
    )

    task = client.prepare_task(
        agent_name="orchestrator",
        prompt="plan work",
        resolved_model_id="kimi-k2.5",
    )
    assert task.sub_agents_enabled is True
    assert task.sub_agents["research-sub"].description == "collect data"
    assert task.metadata["sub_agent_names"] == ["research-sub"]
    assert CREATE_SUB_TASK_TOOL_NAME in task.system_prompt
    assert BATCH_SUB_TASKS_TOOL_NAME in task.system_prompt


def test_sdk_client_unknown_agent_raises_clear_error(tmp_path: Path) -> None:
    client = AgentSDKClient(
        options=AgentSDKOptions(settings_file=Path("local_settings.py"), default_backend="moonshot", workspace=tmp_path),
        agents={"demo": AgentDefinition(description="d", model="m")},
    )
    with pytest.raises(ValueError, match="Unknown agent: missing"):
        client.run_agent(agent_name="missing", prompt="hi")


def test_sdk_query_returns_text_for_completed_run(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "query-ok"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agents={"demo": AgentDefinition(description="helper", model="kimi-k2.5")},
    )

    text = client.query(agent_name="demo", prompt="say ok")
    assert text == "query-ok"


def test_sdk_query_works_for_default_agent(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "default-query"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agent=AgentDefinition(description="helper", model="kimi-k2.5"),
    )

    text = client.query(prompt="say ok")
    assert text == "default-query"


def test_sdk_query_raises_on_non_completed_status(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(steps=[LLMResponse(content="no tools")])
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agents={"demo": AgentDefinition(description="helper", model="kimi-k2.5", max_cycles=1)},
    )

    with pytest.raises(RuntimeError, match="status=max_cycles"):
        client.query(agent_name="demo", prompt="say ok")


def test_sdk_query_can_return_wait_reason_when_not_strict(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="need input",
                    tool_calls=[ToolCall(id="c1", name=ASK_USER_TOOL_NAME, arguments={"question": "pick one"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agents={"demo": AgentDefinition(description="helper", model="kimi-k2.5")},
    )

    text = client.query(agent_name="demo", prompt="say ok", require_completed=False)
    assert "pick one" in text


def test_sdk_query_agent_compatibility_wrapper(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "compat-query"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    client = AgentSDKClient(
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
        agents={"demo": AgentDefinition(description="helper", model="kimi-k2.5")},
    )

    text = client.query_agent(agent_name="demo", prompt="say ok")
    assert text == "compat-query"


def test_sdk_prepare_task_injects_available_skills_into_prompt(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo
description: Demo skill
---
Body
""",
        encoding="utf-8",
    )

    client = AgentSDKClient(
        options=AgentSDKOptions(settings_file=Path("local_settings.py"), default_backend="moonshot", workspace=tmp_path),
        agents={
            "assistant": AgentDefinition(
                description="assist",
                model="kimi-k2.5",
                metadata={"available_skills": [{"location": "demo"}]},
            )
        },
    )

    task = client.prepare_task(
        agent_name="assistant",
        prompt="hello",
        resolved_model_id="kimi-k2.5",
    )

    assert "<available_skills>" in task.system_prompt
    assert "<name>\ndemo\n</name>" in task.system_prompt


def test_sdk_prepare_task_supports_skill_directories_auto_prompt(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    (root / "alpha").mkdir(parents=True)
    (root / "alpha" / "SKILL.md").write_text(
        """---
name: alpha
description: Alpha skill
---
Body
""",
        encoding="utf-8",
    )

    client = AgentSDKClient(
        options=AgentSDKOptions(settings_file=Path("local_settings.py"), default_backend="moonshot", workspace=tmp_path),
        agents={
            "assistant": AgentDefinition(
                description="assist",
                model="kimi-k2.5",
                skill_directories=["skills"],
            )
        },
    )

    task = client.prepare_task(
        agent_name="assistant",
        prompt="hello",
        resolved_model_id="kimi-k2.5",
    )

    assert "<available_skills>" in task.system_prompt
    assert "<name>\nalpha\n</name>" in task.system_prompt
    assert task.metadata["available_skills"] == ["skills"]


def test_sdk_module_level_run_helper(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "module-run"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    run = sdk_run(
        prompt="say ok",
        agent=AgentDefinition(description="helper", model="kimi-k2.5"),
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
    )
    assert run.result.final_answer == "module-run"


def test_sdk_module_level_query_helper(tmp_path: Path) -> None:
    def fake_llm_builder(
        settings_path: str | Path,
        *,
        backend: str,
        model: str,
        timeout_seconds: float = 90.0,
    ) -> tuple[ScriptedLLM, ResolvedModelConfig]:
        del settings_path, timeout_seconds
        llm = ScriptedLLM(
            steps=[
                LLMResponse(
                    content="done",
                    tool_calls=[ToolCall(id="c1", name=TASK_FINISH_TOOL_NAME, arguments={"message": "module-query"})],
                )
            ]
        )
        return llm, _fake_resolved(backend=backend, model=model)

    text = sdk_query(
        prompt="say ok",
        agent=AgentDefinition(description="helper", model="kimi-k2.5"),
        options=AgentSDKOptions(
            settings_file=Path("local_settings.py"),
            default_backend="moonshot",
            workspace=tmp_path,
            llm_builder=fake_llm_builder,
            tool_registry_factory=build_default_registry,
        ),
    )
    assert text == "module-query"
