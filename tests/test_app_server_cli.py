from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vv_agent.app_server.host import AgentResolutionRequest, RunConfigResolutionRequest
from vv_agent.config import EndpointConfig, EndpointOption, ResolvedModelConfig
from vv_agent.model import ModelRef
from vv_agent.model_settings import ModelSettings

_JSON_SCHEMA_NAMES = {
    "AppItem",
    "AppThread",
    "AppTurn",
    "ApprovalDecision",
    "ApprovalRequestParams",
    "ApprovalResolveParams",
    "ClientRequest",
    "InitializeParams",
    "InitializeResponse",
    "JsonRpcMessage",
    "SchemaExportResponse",
    "ServerNotification",
    "ServerRequest",
    "ThreadReadResponse",
    "ThreadResumeResponse",
    "ThreadStartResponse",
    "TurnResumeParams",
    "TurnResumeResponse",
    "TurnStartResponse",
}
_JSON_SCHEMA_FILES = {f"json/{name}.json" for name in _JSON_SCHEMA_NAMES} | {"json/vv_agent_app_server.schemas.json"}


def _generated_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def test_app_server_help_lists_stdio_listener() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vv_agent", "app-server", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "--listen stdio" in result.stdout
    assert "--settings" in result.stdout
    assert "--backend" in result.stdout
    assert "--model" in result.stdout
    assert "--timeout-seconds" in result.stdout
    assert "schema" in result.stdout
    assert "generate-ts" in result.stdout


def test_top_level_help_lists_all_command_surfaces() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vv_agent", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "app-server" in result.stdout
    assert "debug" in result.stdout
    assert "--prompt" in result.stdout


def test_app_server_generate_schema_cli_writes_files(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vv_agent", "app-server", "schema", "--out", str(tmp_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert _generated_files(tmp_path) == _JSON_SCHEMA_FILES
    aggregate = json.loads((tmp_path / "json" / "vv_agent_app_server.schemas.json").read_text(encoding="utf-8"))
    assert set(aggregate) == _JSON_SCHEMA_NAMES
    assert "TurnStartParams" not in aggregate


def test_app_server_generate_ts_cli_writes_self_contained_types(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vv_agent", "app-server", "generate-ts", "--out", str(tmp_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    source = (tmp_path / "ClientRequest.ts").read_text(encoding="utf-8")
    assert "export type ClientRequest" in source
    assert "export interface ApprovalResolveParams" in source
    assert "import " not in source


def test_app_server_start_requires_explicit_model_configuration() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vv_agent", "app-server", "--listen", "stdio"],
        input="",
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "--settings" in result.stderr
    assert "--backend" in result.stderr
    assert "--model" in result.stderr


def test_app_server_production_args_accept_any_order_and_equals_form() -> None:
    from vv_agent import cli

    args = cli._parse_app_server_args(
        [
            "--model=test-model",
            "--timeout-seconds=12.5",
            "--settings",
            "settings.py",
            "--listen=stdio",
            "--backend=test-backend",
        ]
    )

    assert args.listen == "stdio"
    assert args.settings == "settings.py"
    assert args.backend == "test-backend"
    assert args.model == "test-model"
    assert args.timeout_seconds == 12.5


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--settings", "settings.py", "--backend", "b", "--model", "m"], "--listen"),
        (["--listen", "stdio", "--backend", "b", "--model", "m"], "--settings"),
        (["--listen", "stdio", "--settings", "settings.py", "--model", "m"], "--backend"),
        (["--listen", "stdio", "--settings", "settings.py", "--backend", "b"], "--model"),
        (
            [
                "--listen",
                "stdio",
                "--listen=stdio",
                "--settings",
                "settings.py",
                "--backend",
                "b",
                "--model",
                "m",
            ],
            "--listen",
        ),
        (
            [
                "--listen",
                "stdio",
                "--settings",
                "settings.py",
                "--settings=other.py",
                "--backend",
                "b",
                "--model",
                "m",
            ],
            "--settings",
        ),
        (
            [
                "--listen",
                "stdio",
                "--settings",
                "settings.py",
                "--backend",
                "b",
                "--backend=other",
                "--model",
                "m",
            ],
            "--backend",
        ),
        (
            [
                "--listen",
                "stdio",
                "--settings",
                "settings.py",
                "--backend",
                "b",
                "--model",
                "m",
                "--model=other",
            ],
            "--model",
        ),
        (
            [
                "--listen",
                "stdio",
                "--settings",
                "settings.py",
                "--backend",
                "b",
                "--model",
                "m",
                "--timeout-seconds",
                "1",
                "--timeout-seconds=2",
            ],
            "--timeout-seconds",
        ),
        (
            [
                "--listen",
                "tcp",
                "--settings",
                "settings.py",
                "--backend",
                "b",
                "--model",
                "m",
            ],
            "invalid choice",
        ),
        (
            [
                "--listen",
                "stdio",
                "--settings",
                "settings.py",
                "--backend",
                "b",
                "--model",
                "m",
                "--workspace",
                "./workspace",
            ],
            "unrecognized arguments",
        ),
        (
            [
                "--listen",
                "stdio",
                "--settings",
                "settings.py",
                "--backend",
                "b",
                "--model",
                "m",
                "trailing",
            ],
            "unrecognized arguments",
        ),
    ],
)
def test_app_server_production_args_reject_invalid_shapes(argv, expected, capsys) -> None:
    from vv_agent import cli

    assert cli.main(["app-server", *argv]) == 2
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "inf", "-inf", "1e999", "not-a-number"])
def test_app_server_production_args_reject_invalid_timeout(value, capsys) -> None:
    from vv_agent import cli

    exit_code = cli.main(
        [
            "app-server",
            "--listen",
            "stdio",
            "--settings",
            "settings.py",
            "--backend",
            "b",
            "--model",
            "m",
            "--timeout-seconds",
            value,
        ]
    )

    assert exit_code == 2
    assert "--timeout-seconds" in capsys.readouterr().err


def test_app_server_start_resolves_settings_before_reading_stdio(tmp_path) -> None:
    missing_settings = tmp_path / "missing-settings.py"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vv_agent",
            "app-server",
            "--listen",
            "stdio",
            "--settings",
            str(missing_settings),
            "--backend",
            "missing-backend",
            "--model",
            "missing-model",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        returncode = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        pytest.fail("App Server waited for stdin before resolving its settings")
    finally:
        if process.stdin is not None:
            process.stdin.close()
    stderr = process.stderr.read() if process.stderr is not None else ""

    assert returncode == 2
    assert f"Settings file not found: {missing_settings}" in stderr
    assert "Traceback" not in stderr


def test_app_server_runtime_errors_exit_one(monkeypatch, capsys) -> None:
    from vv_agent import cli

    def fail(_argv):
        raise RuntimeError("stdio failed")

    monkeypatch.setattr(cli, "_run_app_server_cli", fail)

    assert cli.main(["app-server"]) == 1
    assert capsys.readouterr().err.strip() == "stdio failed"


def test_app_server_timeout_defaults_to_ninety_seconds() -> None:
    from vv_agent import cli

    args = cli._parse_app_server_args(
        [
            "--listen",
            "stdio",
            "--settings",
            "settings.py",
            "--backend",
            "b",
            "--model",
            "m",
        ]
    )

    assert args.timeout_seconds == 90.0


def test_app_server_configuration_errors_do_not_use_runtime_exit_code(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vv_agent",
            "app-server",
            "--listen",
            "stdio",
            "--settings",
            str(tmp_path / "missing.py"),
            "--backend",
            "b",
            "--model",
            "m",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "Settings file not found" in result.stderr


def test_app_server_host_is_built_from_explicit_model_configuration(monkeypatch, tmp_path) -> None:
    from vv_agent import cli

    endpoint = EndpointConfig(endpoint_id="primary", api_key="key", api_base="https://example.invalid/v1")
    resolved = ResolvedModelConfig(
        backend="test-backend",
        requested_model="test-model",
        selected_model="test-model",
        model_id="provider-model",
        endpoint_options=[EndpointOption(endpoint=endpoint, model_id="provider-model")],
        context_length=256_000,
        function_call_available=True,
    )
    llm = object()
    calls: list[tuple[Path, str, float]] = []

    class Provider:
        def __init__(self, settings_file: Path, backend: str | None = None, timeout_seconds: float = 90.0) -> None:
            self.settings_file = settings_file
            self.backend = backend
            self.timeout_seconds = timeout_seconds

        @classmethod
        def from_settings_file(cls, settings_file: Path) -> Provider:
            return cls(Path(settings_file))

        def with_default_backend(self, backend: str) -> Provider:
            return Provider(self.settings_file, backend, self.timeout_seconds)

        def with_timeout_seconds(self, timeout_seconds: float) -> Provider:
            return Provider(self.settings_file, self.backend, timeout_seconds)

        def resolve(self, model: ModelRef) -> ResolvedModelConfig:
            assert model == ModelRef.named("test-model")
            calls.append((self.settings_file, str(self.backend), self.timeout_seconds))
            return resolved

        def client(self, resolved_model: ResolvedModelConfig):
            assert resolved_model is resolved
            return llm

        def default_settings(self, resolved_model: ResolvedModelConfig) -> ModelSettings:
            assert resolved_model is resolved
            return ModelSettings(timeout_seconds=self.timeout_seconds)

        def default_model_ref(self) -> ModelRef | None:
            return None

    monkeypatch.setattr(cli, "VvLlmModelProvider", Provider)
    settings_file = tmp_path / "settings.py"
    host = cli._build_app_server_host(
        settings_file=settings_file,
        backend="test-backend",
        model="test-model",
        workspace=tmp_path / "workspace",
        max_cycles=4,
        timeout_seconds=12.5,
    )

    agent = host.resolve_agent(AgentResolutionRequest(thread_id="thread_1", agent_key="default"))
    run_config = host.build_run_config(RunConfigResolutionRequest(thread_id="thread_1", agent_key="default"))
    assert run_config.model_provider is not None
    assert run_config.model is not None
    provided_model = run_config.model_provider.resolve(ModelRef.coerce(run_config.model))
    provided_llm = run_config.model_provider.client(provided_model)
    models = host.list_models(__import__("vv_agent.app_server.protocol", fromlist=["ModelListRequest"]).ModelListRequest())

    assert calls == [(settings_file, "test-backend", 12.5), (settings_file, "test-backend", 12.5)]
    assert agent.model == ModelRef.named("test-model")
    assert run_config.max_cycles == 4
    assert run_config.approval_timeout_seconds == 30.0
    assert (provided_llm, provided_model) == (llm, resolved)
    assert models.models[0].id == "provider-model"
    assert models.models[0].context_length == 256_000
    assert models.models[0].supports_tools is True
    assert models.models[0].to_dict()["contextLength"] == 256_000
    assert models.models[0].to_dict()["supportsTools"] is True


def test_debug_app_server_send_message_prints_jsonl_flow() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vv_agent", "debug", "app-server", "send-message", "hello"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert all(message.get("jsonrpc") == "2.0" for message in messages)
    assert any(message.get("method") == "turn/started" for message in messages)
    completed = next(message for message in messages if message.get("method") == "turn/completed")
    assert completed["params"]["status"] == "completed"
    assert not any("error" in message for message in messages)
