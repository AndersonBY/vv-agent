from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import vv_agent.cli as cli
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.types import AgentResult, AgentStatus

CONTRACT_PATH = Path(__file__).with_name("test_cli_contract_v1.json")
CONTRACT_SHA256 = "029e2357351d7029932907bd84448dda92b0dc220eac5f215acd7eb69dee6d09"


def _contract() -> dict[str, Any]:
    payload = CONTRACT_PATH.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == CONTRACT_SHA256
    return json.loads(payload)


def _resolved() -> ResolvedModelConfig:
    return ResolvedModelConfig(
        backend="deepseek",
        requested_model="deepseek-v4-pro",
        selected_model="deepseek-v4-pro",
        model_id="deepseek-v4-pro",
        endpoint_options=[
            EndpointOption(
                endpoint=EndpointConfig(
                    endpoint_id="primary",
                    api_key="test-key",
                    api_base="https://example.invalid/v1",
                ),
                model_id="deepseek-v4-pro",
            )
        ],
        context_length=1_000_000,
        max_output_tokens=384_000,
        function_call_available=True,
        response_format_available=True,
        native_multimodal=True,
    )


def _result(*, status: AgentStatus, error: str | None = None) -> AgentResult:
    return AgentResult(
        status=status,
        messages=[],
        cycles=[],
        final_answer="done" if status == AgentStatus.COMPLETED else None,
        error=error,
    )


def test_cli_contract_fixture_is_reviewable() -> None:
    contract = _contract()

    assert contract["contract"] == "vv-agent-cli-v1"
    assert contract["scope"] == "direct-task"


def test_cli_contract_fixture_matches_rust_copy_when_available() -> None:
    explicit_rust_root = os.environ.get("VV_AGENT_RS_REPO")
    rust_root = Path(explicit_rust_root or Path(__file__).resolve().parents[2] / "vv-agent-rs")
    rust_copy = rust_root / "crates" / "vv-agent" / "tests" / "cli_contract_v1.json"

    if explicit_rust_root is None and not rust_copy.exists():
        return
    assert CONTRACT_PATH.read_bytes() == rust_copy.read_bytes()


def test_settings_file_precedence_uses_explicit_then_primary_then_legacy() -> None:
    environment = {
        "VV_AGENT_LOCAL_SETTINGS": "primary.json",
        "V_AGENT_LOCAL_SETTINGS": "legacy.py",
    }

    explicit = cli._parse_task_args(
        ["--prompt", "task", "--settings-file", "explicit.toml"],
        environ=environment,
    )
    primary = cli._parse_task_args(["--prompt", "task"], environ=environment)
    legacy = cli._parse_task_args(
        ["--prompt", "task"],
        environ={"VV_AGENT_LOCAL_SETTINGS": "  ", "V_AGENT_LOCAL_SETTINGS": "legacy.py"},
    )
    language_default = cli._parse_task_args(["--prompt", "task"], environ={})

    assert explicit.settings_file == "explicit.toml"
    assert primary.settings_file == "primary.json"
    assert legacy.settings_file == "legacy.py"
    assert language_default.settings_file == _contract()["settings_file_resolution"]["language_defaults"]["python"]


def test_multiword_prompt_model_settings_and_resolved_limits_project_to_task() -> None:
    contract = _contract()
    args = cli._parse_task_args(contract["argument_projection"]["argv"], environ={})

    task = cli._build_cli_task(args, _resolved(), task_id="task_fixed")

    expected = contract["argument_projection"]["task"]
    assert task.user_prompt == expected["user_prompt"]
    assert task.max_cycles == expected["max_cycles"]
    assert task.agent_type == expected["agent_type"]
    assert task.model_settings is not None
    assert task.model_settings.to_dict() == expected["model_settings"]
    resolved_projection = contract["resolved_model_projection"]["task"]
    assert task.native_multimodal is resolved_projection["native_multimodal"]
    for key, value in resolved_projection["metadata"].items():
        assert task.metadata[key] == value


@pytest.mark.parametrize(
    ("result", "expected_status", "expected_error"),
    [
        (_result(status=AgentStatus.COMPLETED), "completed", None),
        (_result(status=AgentStatus.FAILED, error="request failed"), "failed", "request failed"),
        (
            _result(status=AgentStatus.FAILED, error="Operation was cancelled"),
            "failed",
            "Operation was cancelled",
        ),
    ],
)
def test_result_json_covers_success_failure_and_cancellation(
    result: AgentResult,
    expected_status: str,
    expected_error: str | None,
) -> None:
    payload = cli._result_payload(result, _resolved())

    assert payload["status"] == expected_status
    assert payload["error"] == expected_error
    assert payload["resolved"] == {
        "backend": "deepseek",
        "selected_model": "deepseek-v4-pro",
        "model_id": "deepseek-v4-pro",
        "endpoint": "primary",
    }


def test_task_cli_returns_clean_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_config(*_args: Any, **_kwargs: Any) -> Any:
        raise cli.ConfigError("settings file not found: missing.json")

    monkeypatch.setattr(cli, "build_openai_llm_from_local_settings", fail_config)

    exit_code = cli._run_task_cli(["--prompt", "inspect", "this", "repository"])
    captured = capsys.readouterr()

    assert exit_code == _contract()["process_outcomes"]["configuration_or_runtime_error"]["exit_code"]
    assert captured.out == ""
    assert captured.err.strip() == "settings file not found: missing.json"
    assert "Traceback" not in captured.err


def test_verbose_runtime_logs_stay_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    handler = cli._build_cli_log_handler(enabled=True)
    assert handler is not None

    handler("run_started", {"task_id": "task_fixed", "model": "test-model", "max_cycles": 1})
    handler(
        "cycle_llm_response",
        {"cycle": 1, "tool_call_names": ["read_file", "write_file"], "assistant_preview": None},
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "[run] start task=task_fixed model=test-model max_cycles=1" in captured.err
    assert 'tool_calls=["read_file","write_file"] assistant=null' in captured.err
    timestamp = captured.err.split("]", 1)[0].removeprefix("[")
    assert timestamp.endswith("Z")
    assert "T" in timestamp


def test_task_cli_process_uses_contract_exit_codes_and_channels() -> None:
    usage = subprocess.run(
        [sys.executable, "-m", "vv_agent.cli"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    missing_path = "/definitely/missing-vv-agent-cli.json"
    configuration = subprocess.run(
        [
            sys.executable,
            "-m",
            "vv_agent.cli",
            "--prompt",
            "inspect",
            "this",
            "repository",
            "--settings-file",
            missing_path,
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    outcomes = _contract()["process_outcomes"]
    assert usage.returncode == outcomes["usage_error"]["exit_code"]
    assert usage.stdout == ""
    assert "--prompt" in usage.stderr
    assert configuration.returncode == outcomes["configuration_or_runtime_error"]["exit_code"]
    assert configuration.stdout == ""
    assert missing_path in configuration.stderr
    assert "Traceback" not in configuration.stderr
