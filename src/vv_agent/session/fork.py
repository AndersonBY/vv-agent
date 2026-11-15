"""Create an independent session from an immutable ended-turn context prefix."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from hashlib import sha256

from vv_agent.app_server.host import AgentResolutionRequest, RunConfigResolutionRequest
from vv_agent.app_server.protocol.identity import client_id
from vv_agent.kernel._content import ContentRef
from vv_agent.memory.manager import MemoryManager
from vv_agent.workspace.streaming import scan_text

from .compaction import memory_settings
from .content_runtime import SESSION_ATTRIBUTES
from .context_view import context_messages, project_view
from .projection import fold_hydrated_records, hydrate_records, hydration_for_session_store
from .records import SessionSpec, copy_json, digest, inherited_seed_policies
from .reducer import _fold_factory_for
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


def _replay(kernel, sid, params):
    from .app_server import SubmissionConflict

    store = kernel.store
    try:
        created = store.read(sid, limit=1).records[0]
    except (KeyError, Conflict):
        return None
    creation = created.record._payload
    attributes = kernel.read_record_content(sid, created.seq, creation["attributes"], slot=SESSION_ATTRIBUTES)
    fork = attributes.get("app_server_fork")
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
    previous = _replay(kernel, sid, params)
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
    created = records[0]
    attributes_ref = created.record._payload["attributes"]
    attributes = kernel.read_record_content(source, created.seq, attributes_ref, slot=SESSION_ATTRIBUTES)
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
    source_records = tuple(r for r in records if r.seq <= cut.seq)
    factory = _fold_factory_for(kernel.store)
    hydration = hydration_for_session_store(kernel.store, source)
    records = hydrate_records(source_records, hydration)
    state = fold_hydrated_records(records, source_records, factory, hydration)
    view = project_view(
        source,
        source_records,
        _fold_factory=factory,
        _hydration=hydration,
    )
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
    messages = context_messages(
        records,
        state,
        definition,
        _fold_factory=factory,
        _source_records=source_records,
    )
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
            "seed": {
                "messages": copied,
                "shared_state": {},
                **inherited_seed_policies(attributes),
            },
            "app_server_fork": fork,
        },
    )
    try:
        with kernel.store.atomic() as tx:
            content_writer = getattr(tx, "content_writer", None)
            if not callable(content_writer):
                raise ValueError("fork requires a transaction with trusted content capability")
            destination = content_writer(tenant_id=kernel.tenant_id, session_id=sid)
            source_content = content_writer(tenant_id=kernel.tenant_id, session_id=source)
            source_principal = page.records[0].record._payload["principal"]
            source_workspace = page.records[0].record._payload["workspace"]
            attributes_labels = ContentRef.from_dict(attributes_ref).labels

            def destination_spec(writer):
                principal_attributes = writer.copy_json_from(source_content, source_principal["attributes"])
                workspace = writer.copy_json_from(source_content, source_workspace)
                return replace(
                    spec,
                    principal=source_principal | {"attributes": principal_attributes.to_dict()},
                    workspace=workspace.to_dict(),
                    attributes=writer.put_json(attributes, labels=attributes_labels).to_dict(),
                )

            receipt = destination.create_session(
                destination_spec,
                consumers=("events", "app_server", "traces"),
            )
    except Conflict:
        previous = _replay(kernel, sid, params)
        if previous is None:
            raise
        return previous
    return {"threadId": sid, "accepted": True, "replayed": receipt.replayed, "lineage": lineage}
