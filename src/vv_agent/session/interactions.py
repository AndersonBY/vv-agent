"""One retained-form builder and validation shared by admission, apply and replay."""

from vv_agent.canonical_json import canonical_json_bytes, canonical_json_sha256
from vv_agent.interaction import InteractionAnswer, InteractionAnswerSpec, InteractionReplyTarget


class InteractionError(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def check_form(form):
    unsigned = {k: v for k, v in form.items() if k not in {"kind", "requestDigest"}}
    if (
        len(form["prompt"].encode("utf-8")) > 16_384
        or len(canonical_json_bytes(form)) > 65_536
        or len(form["answerSpec"]["options"]) > 64
        or any(len(v.encode("utf-8")) > 1024 for v in form["answerSpec"]["options"])
    ):
        raise InteractionError("interaction_form_too_large")
    if form["requestDigest"] != canonical_json_sha256(unsigned):
        raise InteractionError("stale_interaction")


def user_interaction(sid, tid, generation, wait, *, operation_id=None, attempt=None):
    """wait is an op_parked payload or a turn_parked payload, never live config."""
    if operation_id is None:
        prompt, interaction_id, metadata = wait["question"], wait["interaction_id"], {}
    else:
        handle = wait["handle"]
        prompt, interaction_id = handle["question"], handle["interaction_id"]
        metadata = wait.get("interaction_result", {}).get("metadata", {})
    options = metadata.get("options") or []
    try:
        spec = InteractionAnswerSpec(
            metadata.get("selection_type", "single") if options else "text",
            tuple(options),
            bool(metadata.get("allow_custom_options", False)) if options else False,
        )
    except (ValueError, TypeError):
        raise InteractionError("interaction_form_too_large") from None
    unsigned = {
        "sessionId": sid,
        "turnId": tid,
        "generation": generation,
        "interactionId": interaction_id,
        "operationId": operation_id,
        "attempt": attempt,
        "prompt": prompt,
        "answerSpec": spec.to_dict(),
    }
    form = {"kind": "user", **unsigned, "requestDigest": canonical_json_sha256(unsigned)}
    check_form(form)
    return form


def retained_interaction(state, sid, target):
    tid = state.active_turn_id
    if state.closed or state.archived or target.turn_id != tid or tid is None:
        raise InteractionError("stale_interaction")
    turn = state.turns[tid]
    generation = turn.start._payload["generation"]
    if target.operation_id is None:
        if turn.wait is None:
            raise InteractionError("stale_interaction")
        return user_interaction(sid, tid, generation, turn.wait._payload)
    op = state.operations.get(target.operation_id)
    wait = state.waits.get((target.operation_id, target.attempt))
    if op is None or op.turn_id != tid or wait is None or wait["handle"]["kind"] != "user":
        raise InteractionError("stale_interaction")
    return user_interaction(sid, tid, generation, wait, operation_id=target.operation_id, attempt=target.attempt)


def validate_reply(state, sid, target, answer):
    form = retained_interaction(state, sid, target)
    if target.to_dict() != {k: form[k] for k in target.to_dict()}:
        raise InteractionError("stale_interaction")
    try:
        answer.validate(InteractionAnswerSpec.from_dict(form["answerSpec"]))
    except ValueError:
        raise InteractionError("invalid_interaction_answer") from None
    return form


def reply_content(target, answer):
    return {
        "interaction_id": target.interaction_id,
        "operation_id": target.operation_id,
        "text": answer.text if target.operation_id is None and answer.kind == "text" else answer.model_text(),
        "metadata": {"interaction_response": {"target": target.to_dict(), "answer": answer.to_dict()}},
    }


def validate_inbox_reply(state, sid, item):
    """Plain host text binds to an explicit retained id, turn and generation."""
    content = item.payload["content"]
    try:
        if not isinstance(content, dict):
            raise InteractionError("stale_interaction")
        extension = content.get("metadata", {}).get("interaction_response")
        if extension is not None:
            if not isinstance(extension, dict) or set(extension) != {"target", "answer"}:
                raise InteractionError("invalid_interaction_answer")
            target = InteractionReplyTarget.from_dict(extension["target"])
            try:
                answer = InteractionAnswer.from_dict(extension["answer"])
            except (ValueError, TypeError):
                raise InteractionError("invalid_interaction_answer") from None
            if (target.session_id, target.turn_id, target.generation) != (sid, item.target_turn_id, item.generation):
                raise InteractionError("stale_interaction")
            validate_reply(state, sid, target, answer)
            if content != reply_content(target, answer):
                raise InteractionError("invalid_interaction_answer")
        else:
            tid = state.active_turn_id
            if tid is None or item.target_turn_id != tid or item.generation != state.turns[tid].start._payload["generation"]:
                raise InteractionError("stale_interaction")
            matches = []
            turn = state.turns[tid]
            if turn.wait and turn.wait._payload["interaction_id"] == content.get("interaction_id"):
                matches.append(user_interaction(sid, tid, item.generation, turn.wait._payload))
            for (oid, attempt), wait in state.waits.items():
                if (
                    state.operations[oid].turn_id == tid
                    and wait["handle"]["kind"] == "user"
                    and wait["handle"]["interaction_id"] == content.get("interaction_id")
                ):
                    matches.append(user_interaction(sid, tid, item.generation, wait, operation_id=oid, attempt=attempt))
            if len(matches) != 1:
                raise InteractionError("stale_interaction")
            form = matches[0]
            target = InteractionReplyTarget.from_dict(
                {
                    k: form[k]
                    for k in ("sessionId", "turnId", "generation", "interactionId", "operationId", "attempt", "requestDigest")
                }
            )
            if "operation_id" in content and content["operation_id"] != target.operation_id:
                raise InteractionError("stale_interaction")
            answer = InteractionAnswer("text", text=content.get("text"))
            validate_reply(state, sid, target, answer)
        return target, answer
    except InteractionError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError):
        raise InteractionError("invalid_interaction_answer") from None
