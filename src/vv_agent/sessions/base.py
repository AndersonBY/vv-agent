from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Protocol, cast

from vv_agent.checkpoint import canonical_json_bytes
from vv_agent.types import Message, Role

SESSION_COMMIT_SCHEMA = "vv-agent.session-commit.v1"
SESSION_COMMIT_ID_PREFIX = "vv-agent:checkpoint-v2:session:"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SESSION_MESSAGE_FIELDS = frozenset(
    {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "reasoning_content",
        "image_url",
        "metadata",
    }
)
_TOOL_CALL_FIELDS = frozenset({"id", "type", "function", "extra_content"})
_TOOL_FUNCTION_FIELDS = frozenset({"name", "arguments"})


class SessionCommitError(RuntimeError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class Session(Protocol):
    session_id: str

    def get_items(self, limit: int | None = None) -> list[Message]: ...

    def add_items(self, items: list[Message]) -> None: ...

    def add_items_once(
        self,
        commit_id: str,
        payload_digest: str,
        items: list[Message],
    ) -> str: ...

    def pop_item(self) -> Message | None: ...

    def clear(self) -> None: ...

    def clear_session(self) -> None: ...


class SessionStore(Protocol):
    def session(self, session_id: str) -> Session: ...


def session_store_conformance(store: SessionStore, *, session_id: str = "conformance-thread") -> None:
    session = store.session(session_id)
    other_session = store.session(f"{session_id}-other")
    session.clear()
    other_session.clear()

    expected = [
        Message(
            role="user",
            content="inspect the image",
            image_url="data:image/png;base64,AA==",
            metadata={"sequence": 1},
        ),
        Message(
            role="assistant",
            content="",
            name="planner",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"session parity"}'},
                }
            ],
            reasoning_content="Check persistence details.",
            metadata={"sequence": 2},
        ),
        Message(
            role="tool",
            content="result: ok",
            name="lookup",
            tool_call_id="call_1",
            image_url="data:image/png;base64,AQ==",
            metadata={"sequence": 3},
        ),
    ]
    session.add_items(expected)

    commit_items = [Message(role="assistant", content="checkpoint terminal")]
    commit_id = checkpoint_session_commit_id(f"{session_id}/checkpoint")
    payload_digest = session_commit_payload_digest(commit_items)
    if session.add_items_once(commit_id, payload_digest, commit_items) != "committed":
        raise AssertionError("session store did not report the first append-once commit")
    if session.add_items_once(commit_id, payload_digest, commit_items) != "replayed":
        raise AssertionError("session store did not replay an identical append-once commit")
    if session.get_items()[-1:] != commit_items:
        raise AssertionError("session store duplicated or lost append-once items")
    try:
        session.add_items_once(
            commit_id,
            session_commit_payload_digest([Message(role="assistant", content="different")]),
            [Message(role="assistant", content="different")],
        )
    except SessionCommitError as exc:
        if exc.code != "session_commit_identity_conflict":
            raise AssertionError("session store returned the wrong commit conflict code") from exc
    else:
        raise AssertionError("session store accepted a conflicting append-once identity")
    session.pop_item()

    restored = store.session(session_id)
    if restored.get_items() != expected:
        raise AssertionError("session store did not preserve appended messages")
    if restored.get_items(limit=2) != expected[-2:]:
        raise AssertionError("session store limit did not return the newest messages in order")
    if restored.get_items(limit=0) != []:
        raise AssertionError("session store limit=0 must return no messages")
    isolated = restored.get_items()
    isolated[0].content = "mutated outside the store"
    if restored.get_items()[0].content != expected[0].content:
        raise AssertionError("session store leaked mutable message instances")
    if other_session.get_items():
        raise AssertionError("session store did not isolate session ids")
    if restored.pop_item() != expected[-1]:
        raise AssertionError("session store pop_item returned an unexpected message")
    if restored.get_items() != expected[:-1]:
        raise AssertionError("session store pop_item did not remove the message")

    restored.clear_session()
    if session.get_items():
        raise AssertionError("session store clear_session did not clear the session")


def _serialize_message(message: Message) -> str:
    canonical = _normalize_message(message)
    return json.dumps(
        canonical.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def checkpoint_session_commit_id(checkpoint_key: str) -> str:
    if not isinstance(checkpoint_key, str) or not checkpoint_key:
        raise ValueError("checkpoint key must be a non-empty string")
    digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()
    return f"{SESSION_COMMIT_ID_PREFIX}{digest}"


def session_commit_payload(items: list[Message]) -> dict[str, Any]:
    normalized = [_normalize_message(item) for item in items]
    return {
        "schema_version": SESSION_COMMIT_SCHEMA,
        "items": [item.to_dict() for item in normalized],
    }


def session_commit_payload_digest(items: list[Message]) -> str:
    return hashlib.sha256(canonical_json_bytes(session_commit_payload(items), "session commit payload")).hexdigest()


def validate_session_commit(
    commit_id: str,
    payload_digest: str,
    items: list[Message],
) -> list[Message]:
    if not isinstance(commit_id, str) or not commit_id:
        raise SessionCommitError(
            "session commit id must be a non-empty string",
            code="session_commit_identity_invalid",
        )
    if not isinstance(payload_digest, str) or _SHA256_RE.fullmatch(payload_digest) is None:
        raise SessionCommitError(
            "session commit payload digest must be lowercase SHA-256",
            code="session_commit_payload_digest_invalid",
        )
    normalized = [_normalize_message(item) for item in items]
    actual_digest = session_commit_payload_digest(normalized)
    if actual_digest != payload_digest:
        raise SessionCommitError(
            "session commit payload digest does not match items",
            code="session_commit_payload_digest_mismatch",
        )
    return normalized


def _deserialize_message(payload: str | bytes) -> Message:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return _decode_session_item(json.loads(payload))


def _normalize_message(message: Message) -> Message:
    if not isinstance(message, Message):
        raise TypeError("session item must be a Message")
    data: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.name is not None:
        data["name"] = message.name
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    if message.tool_calls is not None:
        data["tool_calls"] = message.tool_calls
    if message.reasoning_content is not None:
        data["reasoning_content"] = message.reasoning_content
    if message.image_url is not None:
        data["image_url"] = message.image_url
    data["metadata"] = message.metadata
    return _decode_canonical_message(data)


def _decode_session_item(value: Any) -> Message:
    return _decode_canonical_message(_expect_object(value, "Message"))


def _decode_canonical_message(data: dict[str, Any]) -> Message:
    _reject_unknown_fields(data, _SESSION_MESSAGE_FIELDS, "Message")
    role = _required_string(data, "role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unknown message role: {role}")
    content = _required_string(data, "content")

    if "tool_calls" not in data:
        tool_calls: list[dict[str, Any]] = []
    else:
        raw_tool_calls = data["tool_calls"]
        if not isinstance(raw_tool_calls, list):
            raise ValueError('"tool_calls" must be an array')
        tool_calls = [_canonical_tool_call(value) for value in raw_tool_calls]

    if "metadata" not in data:
        metadata: dict[str, Any] = {}
    else:
        raw_metadata = data["metadata"]
        if not isinstance(raw_metadata, dict):
            raise ValueError('"metadata" must be an object')
        metadata = cast(dict[str, Any], _canonical_json(raw_metadata, field_name="metadata"))

    return Message(
        role=cast(Role, role),
        content=content,
        name=_optional_string(data, "name"),
        tool_call_id=_optional_string(data, "tool_call_id"),
        tool_calls=tool_calls or None,
        reasoning_content=_optional_string(data, "reasoning_content"),
        image_url=_optional_string(data, "image_url"),
        metadata=metadata,
    )


def _canonical_tool_call(value: Any) -> dict[str, Any]:
    data = _expect_object(value, "ToolCall")
    _reject_unknown_fields(data, _TOOL_CALL_FIELDS, "ToolCall")
    tool_call_id = _required_non_empty_string(data, "id")
    tool_call_type = _required_string(data, "type")
    if tool_call_type != "function":
        raise ValueError(f"unknown tool call type: {tool_call_type}")
    function = _expect_object(data.get("function"), "ToolCall.function")
    _reject_unknown_fields(function, _TOOL_FUNCTION_FIELDS, "ToolCall.function")
    name = _required_non_empty_string(function, "name")
    arguments = _canonical_tool_arguments(_required_string(function, "arguments"))

    canonical: dict[str, Any] = {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        },
    }
    if "extra_content" in data:
        extra_content = data["extra_content"]
        if not isinstance(extra_content, dict):
            raise ValueError('"extra_content" must be an object')
        canonical["extra_content"] = _canonical_json(
            extra_content,
            field_name="extra_content",
        )
    return canonical


def _canonical_tool_arguments(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError('"arguments" must contain a JSON object') from exc
    if not isinstance(decoded, dict):
        raise ValueError('"arguments" must contain a JSON object')
    return cast(dict[str, Any], _canonical_json(decoded, field_name="arguments"))


def _canonical_json(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if value < -(1 << 63) or value > (1 << 64) - 1:
            raise ValueError(f'"{field_name}" contains an integer outside the JSON wire range')
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'"{field_name}" contains a non-finite number')
        return value
    if isinstance(value, list):
        return [_canonical_json(item, field_name=field_name) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f'"{field_name}" object keys must be strings')
        return {key: _canonical_json(value[key], field_name=field_name) for key in sorted(value)}
    raise ValueError(f'"{field_name}" contains a non-JSON value')


def _expect_object(value: Any, type_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{type_name} payload must be an object")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    if key not in data:
        raise ValueError(f'missing required string field "{key}"')
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f'field "{key}" must be a string')
    return value


def _required_non_empty_string(data: dict[str, Any], key: str) -> str:
    value = _required_string(data, key)
    if not value:
        raise ValueError(f'field "{key}" must be a non-empty string')
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f'field "{key}" must be a string')
    return value


def _reject_unknown_fields(data: dict[str, Any], allowed: frozenset[str], type_name: str) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        raise ValueError(f"{type_name} contains unknown fields: {unknown!r}")
