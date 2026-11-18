from __future__ import annotations

import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from vv_agent.kernel._canonical_json import canonical_json_sha256
from vv_agent.runtime.remote_environment import (
    RemoteBindingError,
    RemoteEnvironment,
    RemoteProtocolError,
    RemoteUnsupportedOperation,
    UrllibRemoteTransport,
)
from vv_agent.session.environments import (
    CommandSpec,
    EnvironmentCapabilities,
    IsolationMode,
    LookupBinding,
    LookupStatus,
    NetworkMode,
    StdinMode,
    execution_id_for,
    stable_operation_key,
)
from vv_agent.session.providers import Unknown

TENANT = "tenant-a"
SESSION = "session/a"
OPERATION = "operation/start"
ENVIRONMENT = "environment/remote-a"
PROVIDER = "provider/remote-a"
COMMAND_DIGEST = "3" * 64
NETWORK_REF = "4" * 64
ISOLATION_REF = "5" * 64
NETWORK_POLICY_DIGEST = "6" * 64
ISOLATION_POLICY_DIGEST = "7" * 64


class _Transport:
    def __init__(self, *responses: dict[str, Any] | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, path, deepcopy(payload)))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)


def _capabilities(generation: int = 7) -> EnvironmentCapabilities:
    return EnvironmentCapabilities.create(
        environment_id=ENVIRONMENT,
        generation=generation,
        isolation=IsolationMode.CONTAINER,
        network=NetworkMode.NONE,
        enforces_network_policy=True,
        enforces_resource_limits=True,
        persistent=True,
        environment_capabilities=("command.execute", "filesystem.workspace"),
        network_policy_ref_digest=NETWORK_REF,
        isolation_policy_ref_digest=ISOLATION_REF,
    )


def _policy() -> dict[str, str]:
    return {
        "network_policy_digest": NETWORK_POLICY_DIGEST,
        "isolation_policy_digest": ISOLATION_POLICY_DIGEST,
    }


def _binding(
    *,
    attempt: int = 1,
    request: Any = None,
    operations: str = OPERATION,
) -> LookupBinding:
    request = {"request": 1} if request is None else request
    operation_key = stable_operation_key(SESSION, operations)
    return LookupBinding(
        tenant_id=TENANT,
        session_id=SESSION,
        operation_id=operations,
        attempt=attempt,
        request_digest=canonical_json_sha256(request),
        idempotency_key=operation_key,
        execution_id=execution_id_for(TENANT, ENVIRONMENT, operation_key),
        command_digest=COMMAND_DIGEST,
        environment_id=ENVIRONMENT,
        generation=7,
        provider_id=PROVIDER,
    )


def _command() -> CommandSpec:
    return CommandSpec(
        command_text="printf 'hello\\n'",
        argv=(),
        cwd="/workspace",
        stdin_base64=None,
        stdin_mode=StdinMode.NONE,
        secret_ref_names=(),
        wall_time_ms=30_000,
        output_bytes=64 * 1024,
    )


def _handle(binding: LookupBinding) -> dict[str, Any]:
    return {
        "kind": "provider",
        "provider": PROVIDER,
        "job_id": "job-remote-1",
        "operation_id": binding.operation_id,
        "attempt": binding.attempt,
        "request_digest": binding.request_digest,
        "execution_id": binding.execution_id,
        "command_digest": binding.command_digest,
        "environment_id": binding.environment_id,
        "generation": binding.generation,
        "evidence": ["remote:accepted"],
        "query_ref": "/poll",
        "cancel_ref": "/stop",
    }


def _running_response(binding: LookupBinding) -> dict[str, Any]:
    return {
        "schema_version": "vv-agent.provider-lookup.v1",
        "status": "running",
        "binding": binding.to_dict(),
        "outcome": {
            "kind": "Accepted",
            "handle": _handle(binding),
            "yield_on_steer": False,
        },
        "evidence": ["remote:accepted"],
    }


def _environment(transport: _Transport) -> RemoteEnvironment:
    return RemoteEnvironment(
        transport,
        _capabilities(),
        _policy(),
        provider_id=PROVIDER,
        persistence_verified=True,
    )


def _start(environment: RemoteEnvironment, binding: LookupBinding):
    return environment.start(
        _command(),
        binding,
        expected_capabilities=_capabilities(),
        expected_operational_policy=_policy(),
    )


def test_remote_start_rejects_frozen_capability_and_policy_mismatch_before_transport() -> None:
    binding = _binding()
    transport = _Transport()
    environment = _environment(transport)

    with pytest.raises(RemoteBindingError, match="capabilities"):
        environment.start(
            _command(),
            binding,
            expected_capabilities=_capabilities(generation=8),
            expected_operational_policy=_policy(),
        )
    with pytest.raises(RemoteBindingError, match="operational policy"):
        environment.start(
            _command(),
            binding,
            expected_capabilities=_capabilities(),
            expected_operational_policy=_policy() | {"network_policy_digest": "8" * 64},
        )

    assert transport.calls == []


def test_remote_duplicate_start_returns_the_original_handle() -> None:
    binding = _binding()
    retry_binding = _binding(attempt=2, request={"request": 2})
    transport = _Transport(_running_response(binding), _running_response(retry_binding))
    environment = _environment(transport)

    first = _start(environment, binding)
    duplicate = _start(environment, retry_binding)

    assert first.status is LookupStatus.RUNNING
    assert duplicate.status is LookupStatus.RUNNING
    assert first.outcome is not None and duplicate.outcome is not None
    assert first.outcome.handle["job_id"] == duplicate.outcome.handle["job_id"] == "job-remote-1"
    assert [call[1] for call in transport.calls] == ["/start", "/start"]
    sent = transport.calls[0][2]
    assert sent["binding"]["idempotency_key"] == binding.idempotency_key
    assert sent["binding"]["execution_id"] == binding.execution_id
    assert sent["binding"]["command_digest"] == binding.command_digest
    assert transport.calls[1][2]["binding"]["attempt"] == 2
    assert duplicate.binding == retry_binding
    assert sent["environment"] == _capabilities().to_dict()
    assert sent["operational_policy"] == _policy()


def test_remote_timeout_is_unknown_and_never_starts_the_same_execution_twice() -> None:
    binding = _binding()
    transport = _Transport(TimeoutError("lost response"))
    environment = _environment(transport)

    first = _start(environment, binding)
    retry = _start(environment, binding)

    assert first.status is LookupStatus.UNKNOWN
    assert isinstance(first.outcome, Unknown)
    assert retry.status is LookupStatus.UNKNOWN
    assert retry.evidence == ("remote:start-uncertain",)
    assert [call[1] for call in transport.calls] == ["/start"]


def test_remote_poll_timeout_returns_unknown_without_submitting_start() -> None:
    binding = _binding()
    transport = _Transport(TimeoutError("poll timed out"))
    environment = _environment(transport)

    result = environment.poll(_handle(binding), binding)

    assert result.status is LookupStatus.UNKNOWN
    assert [call[1] for call in transport.calls] == ["/poll"]


def test_remote_stdin_and_stop_target_the_same_accepted_execution() -> None:
    binding = _binding()
    transport = _Transport(_running_response(binding), _running_response(binding))
    environment = _environment(transport)

    environment.write_stdin(_handle(binding), b"hello\n", binding)
    environment.stop(_handle(binding), binding)

    assert [call[1] for call in transport.calls] == ["/stdin", "/stop"]
    assert transport.calls[0][2]["stdin_base64"] == "aGVsbG8K"
    assert transport.calls[1][2]["handle"] == _handle(binding)


def test_remote_response_must_echo_the_exact_authenticated_binding() -> None:
    binding = _binding()
    response = _running_response(binding)
    response["binding"]["generation"] = 8
    transport = _Transport(response)

    with pytest.raises(RemoteProtocolError, match="binding"):
        _start(_environment(transport), binding)


def test_remote_search_rejects_the_undefined_capability_without_transport() -> None:
    transport = _Transport()
    environment = _environment(transport)

    with pytest.raises(RemoteUnsupportedOperation):
        environment.search(query="needle")

    assert transport.calls == []


def test_urllib_transport_posts_json_without_an_http_framework() -> None:
    received: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            size = int(self.headers["Content-Length"])
            received["path"] = self.path
            received["payload"] = json.loads(self.rfile.read(size))
            response = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: Any) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = server.server_address[0]
        port = server.server_address[1]
        transport = UrllibRemoteTransport(f"http://{host}:{port}", timeout_seconds=1)
        result = transport.request("POST", "/probe", {"binding": {"id": "one"}})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result == {"ok": True}
    assert received == {"path": "/probe", "payload": {"binding": {"id": "one"}}}
