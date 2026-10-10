"""Independent provider truth and process barriers, deliberately outside sk_* tables."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from vv_agent.session.providers import Accepted, Definitive, Unknown
from vv_agent.session.records import InboxItem, Record
from vv_agent.tools.base import ToolContext
from vv_agent.types import ToolExecutionResult

from .conftest import Database, open_store


class ControlledProvider:
    def __init__(self, database: Database, mode: str = "definitive", *, idempotent: bool = False):
        self.database, self.mode, self.idempotent = database, mode, idempotent

    def install(self):
        with open_store(self.database) as store, store.atomic():
            store._rows("CREATE TABLE provider_calls (id text PRIMARY KEY, op text, attempt int, key text)")
            store._rows("CREATE TABLE provider_effects (key text PRIMARY KEY)")
            store._rows("""CREATE TABLE provider_jobs (key text PRIMARY KEY, handle text, result text, ready boolean)""")

    def submit(self, plan: Record, *, context: ToolContext):
        key = context.idempotency_key
        with open_store(self.database) as store, store.atomic():
            row = store._rows(
                "INSERT INTO provider_calls(id,op,attempt,key) VALUES (%s,%s,%s,%s) RETURNING id",
                (uuid4().hex, plan.operation_id, plan.attempt, key),
            )[0]
            assert row is not None
            effect_key = key if self.idempotent else f"effect/{row[0]}"
            store._rows("INSERT INTO provider_effects VALUES (%s) ON CONFLICT DO NOTHING", (effect_key,))
            handle = {
                "kind": "provider",
                "provider": plan.payload["provider_binding"],
                "job_id": effect_key,
                "operation_id": plan.operation_id,
                "attempt": plan.attempt,
                "request_digest": plan.payload["request_digest"],
                "evidence": f"receipt/{effect_key}",
                "query_ref": "provider_jobs",
                "cancel_ref": None,
            }
            result = ToolExecutionResult(tool_call_id=plan.payload["request"]["id"], content="provider effect").to_dict()
            store._rows(
                "INSERT INTO provider_jobs VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (effect_key, json.dumps(handle), json.dumps(result), self.mode != "accepted"),
            )
        if self.mode == "accepted":
            return Accepted(handle)
        if self.mode == "unknown":
            return Unknown("provider response lost")
        return Definitive(result, (handle["evidence"],))

    def query(self, handle):
        with open_store(self.database) as store:
            rows = store._rows("SELECT handle,result,ready FROM provider_jobs WHERE key=%s", (handle["job_id"],))
            row = rows[0] if rows else None
        if row is None or json.loads(row[0]) != handle:
            return Unknown("query unavailable")
        return Definitive(json.loads(row[1]), (handle["evidence"],)) if row[2] else Accepted(handle)

    def cancel(self, handle):
        return Unknown("provider has no stop capability")

    def authenticate(self, item: InboxItem, plan: Record) -> bool:
        with open_store(self.database) as store:
            rows = store._rows("SELECT handle,result FROM provider_jobs")
        for handle_json, result_json in rows:
            handle, result = json.loads(handle_json), json.loads(result_json)
            if (
                handle["operation_id"] != plan.operation_id
                or handle["attempt"] != plan.attempt
                or handle["request_digest"] != plan.payload["request_digest"]
            ):
                continue
            if item.kind == "provider_evidence" and item.payload["handle"] == handle:
                return True
            if (
                item.kind == "provider_result"
                and item.payload["result"] == result
                and handle["evidence"] in item.payload["evidence"]
            ):
                return True
        return False

    def counts(self):
        with open_store(self.database) as store:
            calls = store._rows("SELECT count(*) FROM provider_calls")[0]
            effects = store._rows("SELECT count(*) FROM provider_effects")[0]
            assert calls is not None and effects is not None
            return calls[0], effects[0]

    def callback(self, *, input_id="receipt", kind="provider_result", target_turn_id=None) -> InboxItem:
        with open_store(self.database) as store:
            row = store._rows("SELECT handle,result FROM provider_jobs ORDER BY key LIMIT 1")[0]
            assert row is not None
            h, result = map(json.loads, row)
        payload: dict[str, Any] = {
            "operation_id": h["operation_id"],
            "attempt": h["attempt"],
            "request_digest": h["request_digest"],
        }
        if kind == "provider_evidence":
            payload["handle"] = h
        else:
            payload.update(
                provider_binding=h["provider"],
                result=result,
                usage=result.get("raw", {}).get("usage", {}),
                evidence=[h["evidence"]],
            )
        return InboxItem(input_id, kind, payload, target_turn_id=target_turn_id or "s/turn/initial")

    def retain_model_result(self, plan: Record, result: dict[str, Any]):
        key = f"model/{plan.operation_id}/{plan.attempt}"
        handle = {
            "kind": "provider",
            "provider": "model",
            "job_id": key,
            "operation_id": plan.operation_id,
            "attempt": plan.attempt,
            "request_digest": plan.payload["request_digest"],
            "evidence": key,
            "query_ref": "provider_jobs",
            "cancel_ref": None,
        }
        with open_store(self.database) as store, store.atomic():
            store._rows("INSERT INTO provider_jobs VALUES (%s,%s,%s,true)", (key, json.dumps(handle), json.dumps(result)))
