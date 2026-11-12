"""Task value dependencies execute through normal imports without the outer SDK."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("decode_task", [False, True], ids=["core-values", "task-closure"])
def test_task_value_dependencies_without_site_packages(tmp_path, decode_task):
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        from copy import deepcopy
        from pathlib import Path

        root = Path(sys.argv[1])
        decode_task = sys.argv[2] == "True"
        sys.path.insert(0, str(root / "src"))

        class ValueDependenciesOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                if decode_task and fullname == "vv_agent.types":
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, ValueDependenciesOnly())
        import vv_agent
        from vv_agent.kernel._microcompaction_policy import MicrocompactionPolicy, normalize_microcompaction_policy
        from vv_agent.kernel._model_settings import ModelSettings, ResponseFormat, RetrySettings, ToolChoice
        from vv_agent.kernel._prompt_values import PromptBundle, PromptSection
        from vv_agent.kernel._tool_metadata import (
            ToolMetadata, ToolSideEffect, metadata_policy_denial_source, normalize_tool_metadata
        )

        assert Path(vv_agent.__file__) == root / "src/vv_agent/__init__.py"
        assert "Agent" not in vars(vv_agent)
        settings = ModelSettings(
            tool_choice=ToolChoice.tool("lookup"), response_format=ResponseFormat.json_object(),
            retry=RetrySettings(), extra_body={"nested": [1]},
        )
        assert ModelSettings.from_dict(settings.to_dict()) == settings
        settings.to_dict()["extra_body"]["nested"].append(2)
        assert settings.extra_body == {"nested": [1, 2]}
        policy = MicrocompactionPolicy()
        normalized = normalize_microcompaction_policy(policy)
        assert normalized == policy and normalized is not policy

        fixture = json.loads((root / "tests/fixtures/parity/tool_metadata.json").read_text())
        case = fixture["normalization_cases"][0]
        metadata = ToolMetadata.from_dict(case["input"])
        assert metadata.to_dict() == case["expected"]
        cloned = normalize_tool_metadata(metadata)
        assert cloned is not metadata and cloned.to_dict() == metadata.to_dict()
        assert metadata_policy_denial_source(
            metadata, denied_side_effects=[ToolSideEffect.READ]
        ) == "metadata.side_effect"

        source = {"nested": [1]}
        section = PromptSection("policy", "  preserve evidence  ", metadata=source)
        bundle = PromptBundle((section,))
        source["nested"].append(2)
        assert section.to_dict()["metadata"] == {"nested": [1]}
        assert deepcopy(section) is section and deepcopy(bundle) is bundle
        assert PromptBundle.from_dict(bundle.to_dict()) == bundle
        assert bundle.flatten() == "preserve evidence"

        if decode_task:
            from vv_agent import types
            from vv_agent.kernel import _budget_values
            assert types.MAX_WIRE_INTEGER == _budget_values.MAX_WIRE_INTEGER
            assert types.BudgetExhaustion is _budget_values.BudgetExhaustion
            assert types.BudgetUsageSnapshot is _budget_values.BudgetUsageSnapshot
            assert types.ModelSettings is ModelSettings
            assert types.MicrocompactionPolicy is MicrocompactionPolicy
            assert types.PromptBundle is PromptBundle
            vectors = json.loads((root / "tests/fixtures/parity/run_definition.json").read_text())
            for case in vectors["golden_cases"]:
                payload = case["definition"]["task"]
                task = types.AgentTask.from_dict(payload)
                assert task.to_dict() == payload, case["name"]
                assert type(task.prompt_bundle) is PromptBundle
                assert type(task.microcompaction_policy) is MicrocompactionPolicy
                if task.model_settings is not None:
                    assert type(task.model_settings) is ModelSettings

        assert {
            name for name in sys.modules if name.startswith("vv_agent")
            and name not in {"vv_agent", "vv_agent.kernel"}
            and not name.startswith("vv_agent.kernel.")
        } == ({"vv_agent.types"} if decode_task else set())
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(root), str(decode_task)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_dependencies_keep_actual_public_and_outer_objects():
    import vv_agent
    from vv_agent import microcompaction, model_settings, prompt, types
    from vv_agent.kernel import _microcompaction_policy, _model_settings, _prompt_values, _tool_metadata
    from vv_agent.prompt import builder
    from vv_agent.session import records
    from vv_agent.tools import metadata

    for outer, core, names in (
        (model_settings, _model_settings, ("ModelSettings", "ResponseFormat", "RetrySettings", "ToolChoice")),
        (microcompaction, _microcompaction_policy, ("MicrocompactionPolicy", "normalize_microcompaction_policy")),
        (metadata, _tool_metadata, tuple(metadata.__all__)),
        (builder, _prompt_values, ("PromptBundle", "PromptSection")),
    ):
        for name in names:
            value = getattr(core, name)
            assert getattr(outer, name) is value, name
            if name in vv_agent.__all__:
                assert getattr(vv_agent, name) is value, name
    assert prompt.PromptBundle is builder.PromptBundle is types.PromptBundle
    assert prompt.PromptSection is builder.PromptSection
    assert records.ModelSettings is types.ModelSettings is model_settings.ModelSettings
    assert builder.SystemPromptBuilder.__module__ == "vv_agent.prompt.builder"
