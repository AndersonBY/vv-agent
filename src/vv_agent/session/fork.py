"""Create an independent session from an immutable ended-turn context prefix."""

from __future__ import annotations

from contextlib import suppress
from hashlib import sha256

from vv_agent.app_server.host import AgentResolutionRequest, RunConfigResolutionRequest
from vv_agent.app_server.protocol.identity import client_id
from vv_agent.memory.manager import MemoryManager
from vv_agent.workspace.streaming import scan_text

from .compaction import memory_settings
from .context_view import context_messages, project_view
from .records import InboxItem, SessionSpec, copy_json, digest
from .reducer import fold
from .store import Conflict


def validate_params(params):
    if set(params) - {"threadId", "forkId", "throughTurnId", "beforeTurnId", "workspace"}:
        raise ValueError("Unknown thread/fork fields")
    for name in ("threadId", "forkId", "throughTurnId", "beforeTurnId"):
        if (name in params or name in {"threadId", "forkId"}) and (
            not isinstance(params.get(name), str) or not params[name] or len(params[name].encode()) > 512
        ):
            raise ValueError(f"{name} must be a nonempty identity of at most 512 bytes")
    client_id(params["forkId"], "forkId")
    if "throughTurnId" in params and "beforeTurnId" in params:
        raise ValueError("Choose at most one of throughTurnId and beforeTurnId")
    if "workspace" in params and not isinstance(params["workspace"], dict):
        raise ValueError("workspace must be a JSON object")
    digest(params)  # Enforce I-JSON before looking up an identity.


def _replay(store, sid, params):
    from .app_server import SubmissionConflict

    try:
        creation = store.read(sid, limit=1).records[0].record._payload
    except (KeyError, Conflict):
        return None
    fork = creation["attributes"].get("app_server_fork")
    if fork is None or digest(fork["params"]) != digest(params):
        raise SubmissionConflict("submission_conflict")
    return {"threadId": sid, "accepted": True, "replayed": True, "lineage": copy_json(fork["lineage"])}


def _artifact_backend(kernel, host, sid, binding):
    metadata = copy_json(binding["metadata"])
    agent = host.resolve_agent(AgentResolutionRequest(sid, binding["agent_key"], binding["cwd"], metadata))
    config = host.build_run_config(RunConfigResolutionRequest(sid, binding["agent_key"], binding["cwd"], metadata))
    runtime = kernel.runtime(agent, config)
    return runtime.memory_manager.workspace_backend or runtime.config.workspace_backend or runtime.workspace_backend


def fork_thread(kernel, host, params):
    source = params["threadId"]
    sid = "fork/" + digest([source, params["forkId"]])
    previous = _replay(kernel.store, sid, params)
    if previous is not None:
        return previous
    try:
        page = kernel.store.read(source)
    except Conflict as exc:
        if str(exc) == "session or cursor not found":
            raise KeyError(source) from exc
        raise
    records = page.records
    while records[-1].seq < page.head_seq:
        records += kernel.store.read(source, after_seq=records[-1].seq, through_seq=page.head_seq).records
    attributes = records[0].record._payload["attributes"]
    if "app_server" not in attributes:
        raise KeyError(source)
    ended = [r for r in records if r.record.kind == "turn_ended"]
    target = params.get("throughTurnId", params.get("beforeTurnId"))
    if target is not None:
        if not any(r.record.turn_id == target and r.record.kind == "turn_started" for r in records):
            raise ValueError("Unknown fork cut turn")
        terminal = next((r for r in ended if r.record.turn_id == target), None)
        if terminal is None:
            raise ValueError("Fork cut turn has not ended")
        ended = [r for r in ended if r.seq <= terminal.seq and ("beforeTurnId" not in params or r.seq < terminal.seq)]
    cut = ended[-1] if ended else records[0]
    records = tuple(r for r in records if r.seq <= cut.seq)
    state = fold(
        (r.record for r in records),
        consumed_inputs=(InboxItem(**r.record._payload["input"]) for r in records if r.record.kind == "input_applied"),
    )
    view = project_view(source, records)
    definition = copy_json(
        state.turns[next(reversed(state.turns))].start._payload["definition"]
        if state.turns
        else attributes.get("app_server_context")
    )
    if definition is None:
        raise ValueError("Fork source has no retained context definition")
    if state.model_selection is not None:
        selected = state.model_selection._payload
        effective = selected["input"]["payload"]["selection"] | {"binding": selected["model_binding"]}
    elif "app_server_fork" in attributes:
        effective = attributes["app_server_fork"]["effective_model"]
    else:
        effective = {
            "model": definition["task"]["model"],
            "model_settings": definition["task"].get("model_settings") or {},
            "route": {},
            "binding": definition["model_binding"],
        }
    messages = context_messages(records, state, definition)
    copied = [m.to_dict() for m in messages]
    lineage = {
        "sourceThreadId": source,
        "cutTurnId": cut.record.turn_id,
        "sourceRecordSeq": cut.seq,
        "sourceRecordDigest": cut.record.digest,
        "contextDigest": digest(copied),
    }
    binding = copy_json(attributes["app_server"])
    if "workspace" in params:
        binding["metadata"]["workspace"] = copy_json(params["workspace"])
    binding["metadata"]["lineage"] = lineage
    baseline = attributes.get("app_server_context") if "app_server_fork" in attributes else None
    definition["memory_settings"] = memory_settings(baseline or definition, state.model_selection)
    definition["model_binding"] = copy_json(effective["binding"])
    definition["task"]["model"] = effective["model"]
    definition["task"]["model_settings"] = copy_json(effective.get("model_settings") or {})
    definition["task"].update(task_id=sid + "/context", user_prompt="", initial_messages=[], initial_shared_state={})
    definition["task"]["metadata"].pop("vv_session", None)
    definition["task"]["metadata"].pop("_vv_agent_app_server_compaction_only", None)
    # Keep the existing artifact layout; workspace files are exclusively host-owned.
    evidence = MemoryManager().compaction_evidence(messages)
    if evidence["artifacts"]:
        backend = _artifact_backend(kernel, host, sid, binding)
        source_backend = None
        for entry in evidence["artifacts"]:
            ref = entry["artifact_ref"]
            try:
                if not backend.exists(ref["path"]):
                    if source_backend is None:
                        source_backend = _artifact_backend(kernel, host, source, attributes["app_server"])
                    body = source_backend.read_bytes(ref["path"])
                    if len(body) != ref["size_bytes"] or sha256(body).hexdigest() != ref["sha256"]:
                        raise ValueError("Source artifact integrity mismatch")
                    # Another node may have prepared the same immutable artifact.
                    with suppress(FileExistsError):
                        backend.write_text_exclusive(ref["path"], body.decode("utf-8"))
                scanned = scan_text(backend, ref["path"], lambda _text: None)
                intact = scanned.valid_utf8 and scanned.size_bytes == ref["size_bytes"] and scanned.sha256 == ref["sha256"]
            except Exception:
                intact = False
            if not intact:
                raise ValueError("Fork compaction artifact is unavailable or corrupt")
    fork = {
        "params": copy_json(params),
        "lineage": lineage,
        "effective_model": copy_json(effective),
        "last_compaction": view["lastCompaction"],
    }
    spec = SessionSpec(
        sid,
        page.records[0].record._payload["principal"],
        page.records[0].record._payload["workspace"],
        attributes={
            "app_server": binding,
            "app_server_context": definition,
            "seed": {"messages": copied, "shared_state": {}},
            "app_server_fork": fork,
        },
    )
    try:
        with kernel.store.atomic() as tx:
            receipt = tx.create(spec, consumers=("events", "app_server", "traces"))
    except Conflict:
        previous = _replay(kernel.store, sid, params)
        if previous is None:
            raise
        return previous
    return {"threadId": sid, "accepted": True, "replayed": receipt.replayed, "lineage": lineage}
