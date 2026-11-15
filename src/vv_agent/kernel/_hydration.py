"""Typed, dependency-injected hydration for S1 content references.

This module is the kernel-side boundary between encrypted content readers and
runtime values. It does not own a database connection, key, or execution state:
the caller supplies a reader already scoped to one tenant, session, and owner.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol, cast, runtime_checkable

from ._budget_values import RunBudgetLimits
from ._canonical_json import canonical_json_bytes, validate_sha256
from ._content import ContentBindingMismatch, ContentRef
from ._content_store import (
    ContentIntegrityError as _BackendContentIntegrityError,
)
from ._content_store import (
    ContentUnavailable as _BackendContentUnavailable,
)
from ._content_store import (
    KeyUnavailable as _BackendKeyUnavailable,
)
from ._execution_values import AgentTask
from ._message_values import Message
from ._record_codec import RecordError, validate
from ._record_schema import CHILD_ADMISSION, MODEL_BINDING, SEED


class HydrationError(RuntimeError):
    """Base class for failures while turning an encrypted slot into a value."""


class MissingContent(HydrationError):
    """The linked content object does not exist or is no longer readable."""


class MissingContentLink(HydrationError):
    """The tenant/session/owner/slot link is absent or does not name this ref."""


class KeyUnavailable(HydrationError):
    """The session key is unavailable and hydration may be retried later."""


class IntegrityFailure(HydrationError):
    """Content metadata, ciphertext, authentication, or claimed size is invalid."""


class TypedSchemaFailure(HydrationError):
    """The decrypted bytes are not valid current, canonical typed content."""


class ReferenceDigestMismatch(HydrationError):
    """The ContentRef does not match the reference digest recorded by its owner."""


class UnsupportedHydrationSlot(HydrationError):
    """The requested content shape has no complete typed parser in this phase."""


@runtime_checkable
class ContentReader(Protocol):
    """Read one owner-bound content link and return its decrypted bytes.

    Implementations must verify the session, owner, slot, and ``ref`` link
    binding before decrypting. They must raise :class:`MissingContentLink` for
    an absent or mismatched link and :class:`MissingContent` for a present link
    whose content object is unavailable. Key and integrity failures may use
    the content-store exceptions and are translated by ``SessionHydrator``.
    """

    def read(
        self,
        tenant_id: str,
        session_id: str,
        owner: str,
        slot: str,
        ref: ContentRef,
    ) -> bytes: ...


class HydrationSlot(StrEnum):
    CONTENT_REF = "content_ref"
    PRINCIPAL_ATTRIBUTES = "principal.attributes"
    WORKSPACE = "workspace"
    DEFINITION = "definition"
    INPUT = "input"
    TASK = "task"
    BUDGET = "budget"
    REQUEST = "request"
    RESULT = "result"
    USAGE = "usage"
    EVIDENCE = "evidence"
    MESSAGES = "messages"


@dataclass(frozen=True, slots=True)
class DefinitionContent:
    """The current definition container with its embedded task parsed strictly."""

    agent_name: str
    memory_settings: dict[str, Any]
    child_tools: tuple[str, ...]
    model_binding: dict[str, Any]
    capabilities: dict[str, dict[str, Any]]
    task: AgentTask
    tools: tuple[dict[str, Any], ...]


_BUDGET_FIELDS = frozenset(
    {
        "max_total_tokens",
        "max_uncached_input_tokens",
        "max_tool_calls",
        "max_tool_calls_by_name",
        "max_wall_time_ms",
        "max_host_cost",
        "unavailable_metric_policy",
    }
)
_DEFINITION_FIELDS = frozenset(
    {
        "agent_name",
        "memory_settings",
        "child_tools",
        "model_binding",
        "capabilities",
        "task",
        "tools",
    }
)
_HOST_COST_FIELDS = frozenset({"unit", "currency", "amount_microunits"})


def reference_digest(ref: ContentRef) -> str:
    """Return ``SHA256(JCS(ref.to_dict()))`` for a committed reference form.

    ``ContentRef.mac`` authenticates plaintext with the session content key.
    This digest authenticates only the canonical reference object and is kept in
    a separate domain.
    """

    if not isinstance(ref, ContentRef):
        raise TypeError("ref must be ContentRef")
    return sha256(canonical_json_bytes(ref.to_dict(), "ContentRef reference form")).hexdigest()


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _decode_json(body: bytes, slot: HydrationSlot) -> Any:
    if type(body) is not bytes:
        raise TypedSchemaFailure(f"{slot.value} reader must return bytes")
    try:
        text = body.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)
        canonical = canonical_json_bytes(value, slot.value)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise TypedSchemaFailure(f"{slot.value} is not valid canonical JSON") from exc
    if canonical != body:
        raise TypedSchemaFailure(f"{slot.value} is not canonical JSON")
    return value


def _object(value: Any, slot: HydrationSlot) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypedSchemaFailure(f"{slot.value} must be a JSON object")
    return value


def _exact_object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        actual = set(value) if type(value) is dict else set()
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise TypedSchemaFailure(f"{label} fields are invalid: missing={missing}, unknown={unknown}")
    return value


def _validate_shared(value: Any, schema: dict[str, Any], label: str) -> None:
    try:
        validate(value, schema, canonical=True)
    except RecordError as exc:
        raise TypedSchemaFailure(f"{label} does not match its current schema") from exc


def _parse_agent_task(value: Any) -> AgentTask:
    data = _object(value, HydrationSlot.TASK)
    try:
        return AgentTask.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise TypedSchemaFailure("task does not match the current AgentTask schema") from exc


def _parse_definition(value: Any) -> DefinitionContent:
    data = _exact_object(value, _DEFINITION_FIELDS, "definition")
    agent_name = data["agent_name"]
    if not isinstance(agent_name, str) or not agent_name:
        raise TypedSchemaFailure("definition.agent_name must be a non-empty string")
    for field_name in ("memory_settings", "model_binding", "capabilities", "task"):
        if type(data[field_name]) is not dict:
            raise TypedSchemaFailure(f"definition.{field_name} must be a JSON object")
    child_tools = data["child_tools"]
    if type(child_tools) is not list or not all(isinstance(item, str) and item for item in child_tools):
        raise TypedSchemaFailure("definition.child_tools must be an array of non-empty strings")
    if child_tools != sorted(set(child_tools)):
        raise TypedSchemaFailure("definition.child_tools must be sorted and duplicate-free")
    capabilities = data["capabilities"]
    if not all(isinstance(key, str) and key and type(item) is dict for key, item in capabilities.items()):
        raise TypedSchemaFailure("definition.capabilities must map non-empty names to objects")
    tools = data["tools"]
    if type(tools) is not list or not all(type(item) is dict for item in tools):
        raise TypedSchemaFailure("definition.tools must be an array of objects")
    _validate_shared(data["model_binding"], MODEL_BINDING, "definition.model_binding")
    return DefinitionContent(
        agent_name=agent_name,
        memory_settings=data["memory_settings"],
        child_tools=tuple(child_tools),
        model_binding=data["model_binding"],
        capabilities=capabilities,
        task=_parse_agent_task(data["task"]),
        tools=tuple(tools),
    )


def _parse_principal_attributes(value: Any) -> dict[str, Any]:
    data = _object(value, HydrationSlot.PRINCIPAL_ATTRIBUTES)
    if "seed" in data:
        _validate_shared(data["seed"], SEED, "principal attributes seed")
    if "child_admission" in data:
        _validate_shared(data["child_admission"], CHILD_ADMISSION, "principal child admission")
    return data


def _parse_budget(value: Any) -> RunBudgetLimits:
    data = _exact_object(value, _BUDGET_FIELDS, "budget")
    host_cost = data["max_host_cost"]
    if host_cost is not None:
        _exact_object(host_cost, _HOST_COST_FIELDS, "budget.max_host_cost")
    try:
        return RunBudgetLimits.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise TypedSchemaFailure("budget does not match RunBudgetLimits") from exc


def _parse_messages(value: Any) -> list[Message]:
    if type(value) is not list:
        raise TypedSchemaFailure("messages must be a JSON array")
    result: list[Message] = []
    for index, item in enumerate(value):
        if type(item) is not dict:
            raise TypedSchemaFailure(f"messages[{index}] must be an object")
        try:
            result.append(Message.from_dict(cast(dict[str, Any], item)))
        except (TypeError, ValueError) as exc:
            raise TypedSchemaFailure(f"messages[{index}] does not match the current Message schema") from exc
    return result


def _translate_backend_error(exc: Exception) -> HydrationError:
    if isinstance(exc, _BackendContentUnavailable):
        return MissingContent("content is unavailable")
    if isinstance(exc, _BackendKeyUnavailable):
        return KeyUnavailable("session content key is unavailable")
    if isinstance(exc, _BackendContentIntegrityError):
        return IntegrityFailure("content failed integrity validation")
    if isinstance(exc, ContentBindingMismatch):
        return MissingContentLink("content link binding does not match the requested owner or slot")
    return HydrationError("content reader failed")


class SessionHydrator:
    """Hydrate explicitly typed slots from one tenant/session/owner binding."""

    def __init__(self, reader: ContentReader, *, tenant_id: str, session_id: str, owner: str) -> None:
        try:
            self._tenant_id = _non_empty(tenant_id, "tenant_id")
            self._session_id = _non_empty(session_id, "session_id")
            self._owner = _non_empty(owner, "owner")
        except ValueError as exc:
            raise TypedSchemaFailure(str(exc)) from exc
        if not isinstance(reader, ContentReader):
            raise TypeError("reader must implement ContentReader")
        self._reader = reader

    @property
    def binding(self) -> tuple[str, str, str]:
        return self._tenant_id, self._session_id, self._owner

    def _read(self, slot: HydrationSlot, ref: ContentRef, expected_reference_digest: str | None) -> Any:
        if not isinstance(ref, ContentRef):
            raise TypedSchemaFailure(f"{slot.value} requires ContentRef")
        if expected_reference_digest is not None:
            try:
                expected_reference_digest = validate_sha256(
                    expected_reference_digest,
                    f"{slot.value} expected reference digest",
                )
            except ValueError as exc:
                raise TypedSchemaFailure(str(exc)) from exc
            actual = reference_digest(ref)
            if not hmac.compare_digest(actual, expected_reference_digest):
                raise ReferenceDigestMismatch(f"{slot.value} reference digest does not match")
        try:
            body = self._reader.read(self._tenant_id, self._session_id, self._owner, slot.value, ref)
        except (MissingContent, MissingContentLink, KeyUnavailable, IntegrityFailure):
            raise
        except (
            _BackendContentUnavailable,
            _BackendKeyUnavailable,
            _BackendContentIntegrityError,
            ContentBindingMismatch,
        ) as exc:
            raise _translate_backend_error(exc) from exc
        if type(body) is not bytes:
            raise IntegrityFailure(f"{slot.value} reader returned a non-bytes value")
        if len(body) != ref.size:
            raise IntegrityFailure(f"{slot.value} size does not match ContentRef")
        return _decode_json(body, slot)

    def hydrate(
        self,
        slot: HydrationSlot | str,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> Any:
        """Dispatch only statically declared slots and reject all others."""

        try:
            selected = HydrationSlot(slot)
        except ValueError as exc:
            raise UnsupportedHydrationSlot(f"unsupported hydration slot: {slot!r}") from exc
        if selected is HydrationSlot.CONTENT_REF:
            return self.content_ref(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.PRINCIPAL_ATTRIBUTES:
            return self.principal_attributes(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.WORKSPACE:
            return self.workspace(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.DEFINITION:
            return self.definition(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.INPUT:
            return self.input_content(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.TASK:
            return self.task(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.BUDGET:
            return self.budget(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.REQUEST:
            return self.request(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.RESULT:
            return self.result(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.USAGE:
            return self.usage(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.EVIDENCE:
            return self.evidence(ref, expected_reference_digest=expected_reference_digest)
        if selected is HydrationSlot.MESSAGES:
            return self.messages(ref, expected_reference_digest=expected_reference_digest)
        raise UnsupportedHydrationSlot(f"unsupported hydration slot: {selected.value}")

    def content_ref(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> ContentRef:
        value = _object(self._read(HydrationSlot.CONTENT_REF, ref, expected_reference_digest), HydrationSlot.CONTENT_REF)
        try:
            return ContentRef.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise TypedSchemaFailure("content_ref does not match the current ContentRef schema") from exc

    def principal_attributes(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> dict[str, Any]:
        return _parse_principal_attributes(self._read(HydrationSlot.PRINCIPAL_ATTRIBUTES, ref, expected_reference_digest))

    def workspace(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> Any:
        return self._read(HydrationSlot.WORKSPACE, ref, expected_reference_digest)

    def definition(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> DefinitionContent:
        return _parse_definition(self._read(HydrationSlot.DEFINITION, ref, expected_reference_digest))

    def input_content(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> Any:
        """Return one explicitly declared input value without recursing into it."""

        return self._read(HydrationSlot.INPUT, ref, expected_reference_digest)

    def task(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> AgentTask:
        return _parse_agent_task(self._read(HydrationSlot.TASK, ref, expected_reference_digest))

    def budget(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> RunBudgetLimits:
        return _parse_budget(self._read(HydrationSlot.BUDGET, ref, expected_reference_digest))

    def request(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> dict[str, Any]:
        """Return the declared request object without guessing nested C slots."""

        return _object(self._read(HydrationSlot.REQUEST, ref, expected_reference_digest), HydrationSlot.REQUEST)

    def result(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> dict[str, Any]:
        """Return a result envelope; operation-kind decoders remain caller-owned."""

        return _object(self._read(HydrationSlot.RESULT, ref, expected_reference_digest), HydrationSlot.RESULT)

    def usage(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> dict[str, Any]:
        return _object(self._read(HydrationSlot.USAGE, ref, expected_reference_digest), HydrationSlot.USAGE)

    def evidence(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> Any:
        return self._read(HydrationSlot.EVIDENCE, ref, expected_reference_digest)

    def messages(
        self,
        ref: ContentRef,
        *,
        expected_reference_digest: str | None = None,
    ) -> list[Message]:
        return _parse_messages(self._read(HydrationSlot.MESSAGES, ref, expected_reference_digest))


__all__ = [
    "ContentReader",
    "DefinitionContent",
    "HydrationError",
    "HydrationSlot",
    "IntegrityFailure",
    "KeyUnavailable",
    "MissingContent",
    "MissingContentLink",
    "ReferenceDigestMismatch",
    "SessionHydrator",
    "TypedSchemaFailure",
    "UnsupportedHydrationSlot",
    "reference_digest",
]
