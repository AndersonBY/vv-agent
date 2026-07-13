from __future__ import annotations

import json
import math
from typing import Any, Protocol, cast

from vv_agent.types import Message, Role


class Session(Protocol):
    session_id: str

    def get_items(self, limit: int | None = None) -> list[Message]:
        ...

    def add_items(self, items: list[Message]) -> None:
        ...

    def pop_item(self) -> Message | None:
        ...

    def clear(self) -> None:
        ...

    def clear_session(self) -> None:
        ...


class SessionStore(Protocol):
    def session(self, session_id: str) -> Session:
        ...


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
    data = _expect_object(value, "Message")
    if "type" not in data:
        return _decode_canonical_message(data)

    item_type = data.get("type")
    if not isinstance(item_type, str):
        raise ValueError('missing required string field "type"')
    if item_type in {"system", "user", "assistant"}:
        return _decode_canonical_message(
            {
                "role": item_type,
                "content": _required_string(data, "content"),
            }
        )
    if item_type == "tool":
        return _decode_canonical_message(
            {
                "role": "tool",
                "content": _required_string(data, "content"),
                "tool_call_id": _required_string(data, "tool_call_id"),
            }
        )
    if item_type == "message":
        return _decode_canonical_message(_expect_object(data.get("message"), "Message"))
    raise ValueError(f"unknown session item type: {item_type}")


def _decode_canonical_message(data: dict[str, Any]) -> Message:
    role = _required_string(data, "role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unknown message role: {role}")
    raw_content = data.get("content")
    content = raw_content if isinstance(raw_content, str) else ""

    raw_tool_calls = data.get("tool_calls")
    tool_calls = (
        [_canonical_tool_call(value) for value in raw_tool_calls]
        if isinstance(raw_tool_calls, list)
        else []
    )
    raw_metadata = data.get("metadata")
    if raw_metadata is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw_metadata, dict):
        metadata = cast(dict[str, Any], _canonical_json(raw_metadata, field_name="metadata"))
    else:
        raise ValueError('"metadata" must be an object')

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
    function = data.get("function")
    if isinstance(function, dict):
        tool_call_id = _required_string(data, "id")
        name = _required_string(function, "name")
        arguments = _canonical_tool_arguments(function.get("arguments", "{}"))
    else:
        tool_call_id = _required_string(data, "id")
        name = _required_string(data, "name")
        raw_arguments = data.get("arguments")
        if raw_arguments is None:
            arguments = {}
        elif isinstance(raw_arguments, dict):
            arguments = cast(
                dict[str, Any],
                _canonical_json(raw_arguments, field_name="arguments"),
            )
        else:
            raise ValueError('"arguments" must be an object')

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
    extra_content = data.get("extra_content")
    if isinstance(extra_content, dict):
        canonical["extra_content"] = _canonical_json(
            extra_content,
            field_name="extra_content",
        )
    return canonical


def _canonical_tool_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return cast(dict[str, Any], _canonical_json(value, field_name="arguments"))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, dict):
            return cast(dict[str, Any], _canonical_json(decoded, field_name="arguments"))
    return {}


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
        return {
            key: _canonical_json(value[key], field_name=field_name)
            for key in sorted(value)
        }
    raise ValueError(f'"{field_name}" contains a non-JSON value')


def _expect_object(value: Any, type_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{type_name} payload must be an object")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f'missing required string field "{key}"')
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None
