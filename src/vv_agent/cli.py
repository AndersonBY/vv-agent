from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import uuid
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from vv_agent import Agent, RunConfig
from vv_agent.app_server import AppServer, StdioJsonlTransport
from vv_agent.app_server.client import run_debug_message
from vv_agent.app_server.host import DefaultAppServerHost, model_summary
from vv_agent.app_server.schema import generate_json_schema, generate_typescript
from vv_agent.config import ConfigError, ResolvedModelConfig
from vv_agent.events import DiagnosticEvent, RunEvent
from vv_agent.kernel._content import ContentBindingMismatch
from vv_agent.kernel._sql import InputAdmissionProofError, LedgerIntegrityError, SessionNotLive
from vv_agent.kernel._store import DEFAULT_TENANT_ID, Conflict
from vv_agent.model import ModelRef, VvLlmModelProvider
from vv_agent.model_settings import ModelSettings
from vv_agent.prompt import build_system_prompt_bundle
from vv_agent.session.ledger import Ed25519LedgerVerifier
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.session.surfaces import _read_snapshot
from vv_agent.types import AgentResult, AgentTask

_MAX_U32 = (1 << 32) - 1
_APP_SERVER_DEFAULT_TIMEOUT_SECONDS = 90.0
_APP_SERVER_APPROVAL_TIMEOUT_SECONDS = 30.0
_APP_SERVER_DEFAULT_WORKSPACE = Path("./workspace")
_APP_SERVER_DEFAULT_MAX_CYCLES = 80
_LEDGER_OUTPUT_VERSION = 1
_LEDGER_SESSION_PAGE_SIZE = 256


class _AppServerCliError(ValueError):
    exit_code = 2


class _AppServerArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _AppServerCliError(f"{self.format_usage().rstrip()}\n{self.prog}: error: {message}")


class _StoreUnique(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        if hasattr(namespace, self.dest):
            parser.error(f"argument {option_string}: may only be specified once")
        setattr(namespace, self.dest, values)


class _LedgerCliUsageError(ValueError):
    pass


class _LedgerPublicKeyError(ValueError):
    pass


class _LedgerArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _LedgerCliUsageError(f"{self.format_usage().rstrip()}\n{self.prog}: error: {message}")


class _TaskCliUsageError(ValueError):
    pass


class _TaskArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _TaskCliUsageError(f"{message}\n\n{self.format_help().rstrip()}")


def _build_cli_event_handler(*, enabled: bool):
    if not enabled:
        return None

    def handler(event: RunEvent) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        payload = event.to_dict()
        if isinstance(event, DiagnosticEvent):
            details_json = json.dumps(event.details, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            print(f"[{now}] [diagnostic:{event.level}] {event.code} {details_json}", file=sys.stderr, flush=True)
            return
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        print(f"[{now}] [{event.type}] {payload_json}", file=sys.stderr, flush=True)

    return handler


def _parse_non_blank_app_server_value(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def _parse_app_server_timeout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _parse_ledger_sqlite_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    if value == ":memory:":
        raise argparse.ArgumentTypeError("must be a SQLite file path")
    return Path(value)


def _parse_ledger_expected_head(value: str) -> tuple[int, str]:
    seq_text, separator, digest = value.partition(",")
    if not separator or not seq_text or not digest:
        raise argparse.ArgumentTypeError("must use SEQ,DIGEST")
    try:
        seq = int(seq_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sequence must be a non-negative integer") from exc
    if seq < 0:
        raise argparse.ArgumentTypeError("sequence must be a non-negative integer")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hexadecimal characters")
    return seq, digest


def _parse_ledger_public_key(value: str) -> tuple[str, Path]:
    key_id, separator, path = value.partition("=")
    if not separator or not key_id.strip() or not path.strip():
        raise argparse.ArgumentTypeError("must use KEY_ID=PEM_PATH")
    return key_id, Path(path)


def _load_ledger_verifier(public_key_paths: dict[str, Path]) -> Ed25519LedgerVerifier:
    keys = {}
    for key_id, path in public_key_paths.items():
        try:
            if not path.is_file():
                raise OSError("not a file")
            key = load_pem_public_key(path.read_bytes())
        except (OSError, ValueError, UnsupportedAlgorithm):
            raise _LedgerPublicKeyError("could not read an Ed25519 public key from the supplied PEM file") from None
        if not isinstance(key, Ed25519PublicKey):
            raise _LedgerPublicKeyError("supplied public key must be Ed25519")
        keys[key_id] = key
    return Ed25519LedgerVerifier(keys)


def _build_app_server_parser() -> _AppServerArgumentParser:
    parser = _AppServerArgumentParser(
        prog="vv-agent app-server",
        description="Run the vv-agent App Server. Use --listen stdio for JSONL over stdin/stdout.",
        epilog=("Schema commands: schema and generate-ts. Model settings are resolved before the server starts."),
    )
    parser.add_argument(
        "--listen",
        choices=["stdio"],
        required=True,
        default=argparse.SUPPRESS,
        action=_StoreUnique,
        metavar="stdio",
        help="Listen transport. Only stdio JSONL is supported.",
    )
    parser.add_argument(
        "--settings",
        required=True,
        default=argparse.SUPPRESS,
        action=_StoreUnique,
        type=_parse_non_blank_app_server_value,
        help="Path to the LLM settings file",
    )
    parser.add_argument(
        "--backend",
        required=True,
        default=argparse.SUPPRESS,
        action=_StoreUnique,
        type=_parse_non_blank_app_server_value,
        help="Provider backend key in LLM_SETTINGS",
    )
    parser.add_argument(
        "--model",
        required=True,
        default=argparse.SUPPRESS,
        action=_StoreUnique,
        type=_parse_non_blank_app_server_value,
        help="Model key in the selected backend",
    )
    parser.add_argument(
        "--timeout-seconds",
        default=argparse.SUPPRESS,
        action=_StoreUnique,
        type=_parse_app_server_timeout,
        help=f"Provider request timeout (default: {_APP_SERVER_DEFAULT_TIMEOUT_SECONDS:g})",
    )
    return parser


def _parse_app_server_args(argv: list[str]) -> argparse.Namespace:
    args = _build_app_server_parser().parse_args(argv)
    if not hasattr(args, "timeout_seconds"):
        args.timeout_seconds = _APP_SERVER_DEFAULT_TIMEOUT_SECONDS
    return args


def _run_app_server_cli(argv: list[str]) -> None:
    if argv and argv[0] in {"schema", "generate-ts"}:
        parser = argparse.ArgumentParser(prog=f"vv-agent app-server {argv[0]}")
        parser.add_argument("--out", required=True, help="Directory for generated protocol files")
        args = parser.parse_args(argv[1:])
        if argv[0] == "generate-ts":
            generate_typescript(args.out)
        else:
            generate_json_schema(args.out)
        print(args.out)
        return

    args = _parse_app_server_args(argv)
    try:
        host = _build_app_server_host(
            settings_file=Path(args.settings),
            backend=args.backend,
            model=args.model,
            workspace=_APP_SERVER_DEFAULT_WORKSPACE,
            max_cycles=_APP_SERVER_DEFAULT_MAX_CYCLES,
            timeout_seconds=args.timeout_seconds,
        )
    except (ConfigError, OSError, SyntaxError, ValueError) as exc:
        raise _AppServerCliError(str(exc)) from exc
    AppServer(transport=StdioJsonlTransport(), host=host).run_forever()


def _build_app_server_host(
    *,
    settings_file: Path,
    backend: str,
    model: str,
    workspace: Path,
    max_cycles: int,
    timeout_seconds: float = _APP_SERVER_DEFAULT_TIMEOUT_SECONDS,
) -> DefaultAppServerHost:
    provider = (
        VvLlmModelProvider.from_settings_file(settings_file).with_default_backend(backend).with_timeout_seconds(timeout_seconds)
    )
    model_ref = ModelRef.named(model)
    resolved = provider.resolve(model_ref)

    return DefaultAppServerHost(
        agent=Agent(
            name="assistant",
            instructions="You are the vv-agent App Server assistant. Complete user requests with available tools.",
            model=model_ref,
        ),
        run_config=RunConfig(
            model=model_ref,
            model_provider=provider,
            workspace=workspace,
            max_cycles=max_cycles,
            approval_timeout_seconds=_APP_SERVER_APPROVAL_TIMEOUT_SECONDS,
        ),
        models=[model_summary(resolved)],
    )


def _build_ledger_verify_parser() -> _LedgerArgumentParser:
    parser = _LedgerArgumentParser(
        prog="vv-agent ledger verify",
        description="Verify a local S1 SQLite ledger using trusted public keys or explicit hash-only mode.",
    )
    parser.add_argument(
        "--sqlite",
        required=True,
        type=_parse_ledger_sqlite_path,
        help="Path to an existing S1 SQLite session store",
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--session",
        default=argparse.SUPPRESS,
        action=_StoreUnique,
        type=_parse_non_blank_app_server_value,
        help="Session ID in the default tenant",
    )
    scope.add_argument(
        "--tenant",
        default=argparse.SUPPRESS,
        action=_StoreUnique,
        type=_parse_non_blank_app_server_value,
        help="Verify every live session visible through this tenant binding",
    )
    parser.add_argument(
        "--expected-head",
        default=argparse.SUPPRESS,
        action=_StoreUnique,
        type=_parse_ledger_expected_head,
        metavar="SEQ,DIGEST",
        help="Independent trusted head anchor; valid only with --session",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--allow-unsigned",
        action="store_true",
        help="Allow local hash-only verification only when the target scope has no signed evidence",
    )
    mode.add_argument(
        "--public-key",
        default=[],
        action="append",
        type=_parse_ledger_public_key,
        metavar="KEY_ID=PEM_PATH",
        help="Trusted Ed25519 public key in PEM format. Repeat for retained key IDs.",
    )
    return parser


def _parse_ledger_verify_args(argv: list[str]) -> argparse.Namespace:
    parser = _build_ledger_verify_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "tenant") and hasattr(args, "expected_head"):
        parser.error("--expected-head requires --session and cannot be used with --tenant")
    public_keys = dict(args.public_key)
    if len(public_keys) != len(args.public_key):
        parser.error("--public-key key IDs must be unique")
    args.public_key = public_keys
    return args


def _ledger_result_payload(args: argparse.Namespace) -> dict[str, Any]:
    session_id = getattr(args, "session", None)
    tenant_id = getattr(args, "tenant", None) or DEFAULT_TENANT_ID
    expected_present = hasattr(args, "expected_head")
    authenticity = "hash-only" if args.allow_unsigned else "signed"
    if session_id is None:
        checked_scope: dict[str, Any] = {
            "kind": "tenant",
            "tenant_id": tenant_id,
            "session_count": None,
            "sessions_verified": 0,
        }
    else:
        checked_scope = {
            "kind": "session",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "session_count": 1,
            "sessions_verified": 0,
        }
    return {
        "schema_version": _LEDGER_OUTPUT_VERSION,
        "command": "ledger.verify",
        "ok": False,
        "backend": "sqlite",
        "checked_scope": checked_scope,
        "observed_heads": [],
        "authenticity": authenticity,
        "expected_head_present": expected_present,
        "suffix_completeness": False,
        "error": None,
    }


def _ledger_error_object(exc: Exception) -> dict[str, str]:
    message = str(exc).strip() or type(exc).__name__
    if isinstance(exc, _LedgerPublicKeyError):
        return {"code": "invalid_public_key", "message": message}
    if isinstance(exc, ContentBindingMismatch):
        return {"code": "expected_head_mismatch", "message": message}
    if isinstance(exc, InputAdmissionProofError):
        return {"code": "input_proof_error", "message": message}
    if isinstance(exc, LedgerIntegrityError):
        return {"code": "ledger_integrity_error", "message": message}
    if isinstance(exc, SessionNotLive):
        return {"code": "session_not_live", "message": message}
    if isinstance(exc, Conflict):
        code = "session_not_found" if message == "session or cursor not found" else "ledger_conflict"
        return {"code": code, "message": message}
    if isinstance(exc, sqlite3.Error):
        return {"code": "store_unavailable", "message": "SQLite store cannot be opened or read"}
    if isinstance(exc, OSError):
        return {"code": "store_unavailable", "message": "SQLite store cannot be opened or read"}
    if isinstance(exc, ValueError):
        return {"code": "store_incompatible", "message": message}
    return {
        "code": "ledger_verification_failed",
        "message": f"ledger verification failed with {type(exc).__name__}",
    }


def _ledger_observed_head(store: Any, session_id: str) -> dict[str, Any]:
    row = store._head(session_id)
    seq = row[0]
    return {
        "session_id": session_id,
        "seq": seq,
        "digest": store._chain_anchor(session_id, seq),
    }


def _tenant_sessions(store: Any) -> tuple[str, ...]:
    sessions: list[str] = []
    after: str | None = None
    while True:
        page = store.list_sessions(limit=_LEDGER_SESSION_PAGE_SIZE, after=after)
        sessions.extend(page)
        if len(page) < _LEDGER_SESSION_PAGE_SIZE:
            return tuple(sessions)
        after = page[-1]


def _has_signed_evidence(store: Any, sessions: tuple[str, ...]) -> bool:
    for session_id in sessions:
        row = store._rows(
            "SELECT EXISTS(SELECT 1 FROM sk_checkpoint WHERE tenant_id=%s AND session_id=%s) "
            "OR EXISTS(SELECT 1 FROM sk_inbox WHERE tenant_id=%s AND session_id=%s AND "
            "(admission_key_id IS NOT NULL OR admission_signature IS NOT NULL OR admission_signed_ms IS NOT NULL))",
            (store.tenant_id, session_id, store.tenant_id, session_id),
        )
        if bool(row[0][0]):
            return True
    return False


def _populate_ledger_verification(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    path = args.sqlite
    if not path.is_file():
        payload["error"] = {
            "code": "store_not_found",
            "message": "SQLite store path does not identify an existing file",
        }
        return

    tenant_id = payload["checked_scope"]["tenant_id"]
    session_id = getattr(args, "session", None)
    expected_head = getattr(args, "expected_head", None)
    verifier = _load_ledger_verifier(args.public_key) if not args.allow_unsigned else None
    with SQLiteStore.standalone(str(path)) as root, _read_snapshot(root):
        root.connection.execute("PRAGMA query_only=ON")
        store = root.for_tenant(tenant_id)
        sessions = (session_id,) if session_id is not None else _tenant_sessions(store)
        payload["checked_scope"]["session_count"] = len(sessions)
        if args.allow_unsigned and _has_signed_evidence(store, sessions):
            payload["authenticity"] = "signed"
            payload["error"] = {
                "code": "signed_evidence_requires_host_verifier",
                "message": (
                    "signed ledger evidence was detected; a host-configured verifier is required and hash-only "
                    "verification was not performed"
                ),
            }
            return

        root.configure_input_admission(verifier, allow_unsigned=args.allow_unsigned)

        for current_session_id in sessions:
            try:
                seq, digest = store.verify_ledger(
                    current_session_id,
                    expected_head=expected_head if current_session_id == session_id else None,
                )
            except Exception as exc:
                payload["error"] = _ledger_error_object(exc)
                with suppress(Exception):
                    payload["observed_heads"].append(_ledger_observed_head(store, current_session_id))
                return
            payload["observed_heads"].append(
                {
                    "session_id": current_session_id,
                    "seq": seq,
                    "digest": digest,
                }
            )
            payload["checked_scope"]["sessions_verified"] += 1


def _run_ledger_cli(argv: list[str]) -> int:
    if not argv or argv[0] != "verify":
        raise _LedgerCliUsageError("Supported ledger command: vv-agent ledger verify")
    args = _parse_ledger_verify_args(argv[1:])
    payload = _ledger_result_payload(args)
    if not args.allow_unsigned and not args.public_key:
        payload["error"] = {
            "code": "unconfigured_verifier",
            "message": (
                "signed ledger verification requires --public-key KEY_ID=PEM_PATH. "
                "Use --allow-unsigned only for local SQLite hash-only verification."
            ),
        }
    else:
        try:
            _populate_ledger_verification(args, payload)
        except Exception as exc:
            payload["error"] = _ledger_error_object(exc)

    payload["ok"] = payload["error"] is None
    payload["suffix_completeness"] = payload["ok"] and payload["expected_head_present"]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def _print_top_level_help() -> None:
    parser = argparse.ArgumentParser(
        prog="vv-agent",
        description="Run an agent task, host the App Server, verify a ledger, or use debug utilities.",
        epilog=(
            "Commands:\n"
            "  app-server   Run stdio App Server or generate protocol bindings.\n"
            "  debug        Run App Server debug utilities.\n"
            "  ledger       Verify a local S1 SQLite ledger.\n\n"
            "Task mode keeps the existing top-level --prompt interface."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompt", nargs="+", help="Task prompt for direct task mode")
    parser.add_argument("--backend", help="Provider backend key for direct task mode")
    parser.add_argument("--model", help="Model key for direct task mode")
    parser.add_argument("--settings-file", help="Path to local model settings")
    parser.add_argument("--workspace", help="Workspace directory")
    parser.add_argument("--max-cycles", help="Maximum runtime cycles")
    parser.add_argument("--language", help="System prompt language")
    parser.add_argument("--agent-type", help="Agent type")
    parser.add_argument("--temperature", help="Model sampling temperature")
    parser.add_argument("--top-p", help="Model nucleus sampling threshold")
    parser.add_argument("--max-tokens", help="Maximum generated tokens")
    parser.add_argument("--verbose", action="store_true", help="Show runtime logs on stderr")
    parser.print_help()


def _run_debug_cli(argv: list[str]) -> None:
    if len(argv) >= 3 and argv[0] == "app-server" and argv[1] == "send-message":
        for message in run_debug_message(" ".join(argv[2:])):
            print(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
        return
    parser = argparse.ArgumentParser(prog="vv-agent debug")
    parser.add_argument("args", nargs="*")
    parser.parse_args(argv)
    raise SystemExit("Supported debug command: vv-agent debug app-server send-message <message>")


def _non_blank_environment_value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None or not value.strip():
        return None
    return value


def _default_task_settings_file(environ: Mapping[str, str]) -> str:
    return _non_blank_environment_value(environ, "VV_AGENT_LOCAL_SETTINGS") or "local_settings.py"


def _parse_u32(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0 or parsed > _MAX_U32:
        raise argparse.ArgumentTypeError(f"must be between 0 and {_MAX_U32}")
    return parsed


def _parse_positive_u32(value: str) -> int:
    parsed = _parse_u32(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _parse_temperature(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number at least 0") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number at least 0")
    return parsed


def _parse_top_p(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return parsed


def _build_task_parser(*, environ: Mapping[str, str]) -> _TaskArgumentParser:
    parser = _TaskArgumentParser(
        prog="vv-agent",
        description="Run a vv-agent task against configured LLM endpoint",
    )
    parser.add_argument("--prompt", nargs="+", required=True, help="Task prompt")
    parser.add_argument("--backend", default="moonshot", help="Provider backend key in LLM_SETTINGS")
    parser.add_argument("--model", default="kimi-k3", help="Model key in provider models")
    parser.add_argument(
        "--settings-file",
        default=_default_task_settings_file(environ),
        help="Path to local settings (default: VV_AGENT_LOCAL_SETTINGS or local_settings.py)",
    )
    parser.add_argument("--workspace", default="./workspace", help="Workspace directory")
    parser.add_argument("--max-cycles", type=_parse_u32, default=80, help="Max runtime cycles")
    parser.add_argument("--language", default="zh-CN", help="System prompt language (en-US / zh-CN)")
    parser.add_argument("--agent-type", default=None, help="Agent type, e.g. computer")
    parser.add_argument("--temperature", type=_parse_temperature, default=None, help="Model sampling temperature")
    parser.add_argument("--top-p", type=_parse_top_p, default=None, help="Model nucleus sampling threshold")
    parser.add_argument("--max-tokens", type=_parse_positive_u32, default=None, help="Maximum generated tokens")
    parser.add_argument("--verbose", action="store_true", help="Show per-cycle runtime logs")
    return parser


def _parse_task_args(
    argv: list[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> argparse.Namespace:
    parser = _build_task_parser(environ=os.environ if environ is None else environ)
    args = parser.parse_args(argv)
    args.prompt = " ".join(args.prompt)
    if not args.prompt.strip():
        parser.error("--prompt is required")
    args.max_cycles = max(args.max_cycles, 1)
    if args.agent_type is not None and not args.agent_type.strip():
        args.agent_type = None
    return args


def _task_model_settings(args: argparse.Namespace) -> ModelSettings | None:
    if args.temperature is None and args.top_p is None and args.max_tokens is None:
        return None
    return ModelSettings(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def _build_cli_task(
    args: argparse.Namespace,
    resolved: ResolvedModelConfig,
    *,
    task_id: str,
) -> AgentTask:
    prompt_bundle = build_system_prompt_bundle(
        "You are Vector Vein agent runtime demo. Execute tasks with reliable tool usage and clear final outputs.",
        language=args.language,
        allow_interruption=True,
        use_workspace=True,
        enable_todo_management=True,
        agent_type=args.agent_type,
        workspace=Path(args.workspace),
    )
    metadata: dict[str, Any] = {
        "language": args.language,
        "function_call_available": resolved.function_call_available,
        "response_format_available": resolved.response_format_available,
        "native_multimodal": resolved.native_multimodal,
    }
    if resolved.context_length is not None:
        metadata["model_context_window"] = resolved.context_length
    if resolved.max_output_tokens is not None:
        metadata["model_max_output_tokens"] = resolved.max_output_tokens

    return AgentTask(
        task_id=task_id,
        model=resolved.model_id,
        prompt_bundle=prompt_bundle,
        user_prompt=args.prompt,
        max_cycles=max(args.max_cycles, 1),
        agent_type=args.agent_type,
        native_multimodal=resolved.native_multimodal,
        model_settings=_task_model_settings(args),
        metadata=metadata,
    )


def _result_payload(result: AgentResult, resolved: ResolvedModelConfig) -> dict[str, Any]:
    endpoint = resolved.endpoint_options[0].endpoint.endpoint_id if resolved.endpoint_options else None
    return {
        "status": result.status.value,
        "final_answer": result.final_answer,
        "wait_reason": result.wait_reason,
        "error": result.error,
        "cycles": len(result.cycles),
        "todo_list": result.todo_list,
        "resolved": {
            "backend": resolved.backend,
            "selected_model": resolved.selected_model,
            "model_id": resolved.model_id,
            "endpoint": endpoint,
        },
    }


def _run_task_cli(argv: list[str], *, store=None, session_id: str = "cli") -> int:
    try:
        args = _parse_task_args(argv)

        provider = VvLlmModelProvider.from_settings_file(Path(args.settings_file)).with_default_backend(args.backend)
        model_ref = ModelRef.named(args.model)
        resolved = provider.resolve(model_ref)

        task = _build_cli_task(args, resolved, task_id=f"task_{uuid.uuid4().hex[:8]}")
        from vv_agent.session.surfaces import SessionDriver

        driver = SessionDriver(store=store)
        driver.create(session_id, args.workspace)
        handle = driver.start(
            session_id,
            Agent("cli", task.prompt_bundle, model=args.model),
            RunConfig(
                model_provider=provider,
                workspace=Path(args.workspace),
                max_cycles=task.max_cycles,
                stream=_build_cli_event_handler(enabled=args.verbose),
            ),
            args.prompt,
            task=task,
        )
        result = handle.result().raw_result
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        message = str(exc).strip() or type(exc).__name__
        print(message, file=sys.stderr)
        return 1

    print(json.dumps(_result_payload(result, resolved), ensure_ascii=False, indent=2))
    return 0


def _main(args: list[str]) -> int:
    if args and args[0] in {"-h", "--help"}:
        _print_top_level_help()
        return 0
    if args and args[0] == "app-server":
        try:
            _run_app_server_cli(args[1:])
        except _AppServerCliError as exc:
            print(str(exc), file=sys.stderr)
            return exc.exit_code
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            message = str(exc).strip() or type(exc).__name__
            print(message, file=sys.stderr)
            return 1
        return 0
    if args and args[0] == "debug":
        _run_debug_cli(args[1:])
        return 0
    if args and args[0] == "ledger":
        try:
            return _run_ledger_cli(args[1:])
        except _LedgerCliUsageError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            return 130
    return _run_task_cli(args)


def main(argv: list[str] | None = None) -> int:
    exit_code = _main(list(sys.argv[1:] if argv is None else argv))
    if argv is None:
        raise SystemExit(exit_code)
    return exit_code


if __name__ == "__main__":
    main()
