"""Single interaction/intent/identity owners, including ordinary core-only imports."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_interaction_dependencies_execute_without_sdk(tmp_path, flags):
    script = r"""
import importlib.abc, sys

sys.path.insert(0, sys.argv[1])


class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert (
            fullname in {"vv_agent", "vv_agent.kernel"}
            or fullname.startswith("vv_agent.kernel.")
            or fullname.split(".")[0] in sys.stdlib_module_names
        ), fullname


sys.meta_path.insert(0, CoreOnly())
from vv_agent.kernel._interaction_values import (
    InteractionAnswer,
    InteractionAnswerSpec,
    InteractionReplyTarget,
    AppServerUserInteraction,
)
from vv_agent.kernel._interactions import user_interaction, reply_content, validate_inbox_reply, InteractionError
from vv_agent.kernel._summary_requests import compaction_request
from vv_agent.kernel._tool_result_checks import provider_result_identity
from vv_agent.kernel._records import InboxItem
from types import SimpleNamespace

form = user_interaction("s", "t", 1, {"question": "雪 / ~", "interaction_id": "wait"})
target = InteractionReplyTarget.from_dict(
    {k: form[k] for k in ("sessionId", "turnId", "generation", "interactionId", "operationId", "attempt", "requestDigest")}
)
assert AppServerUserInteraction.from_dict(form).to_dict() == form
answer = InteractionAnswer("text", text="answer")
state = SimpleNamespace(
    active_turn_id="t",
    closed=False,
    archived=False,
    turns={
        "t": SimpleNamespace(
            start=SimpleNamespace(_payload={"generation": 1}),
            wait=SimpleNamespace(_payload={"question": "雪 / ~", "interaction_id": "wait"}),
        )
    },
    waits={},
)
item = InboxItem("reply", "user", {"content": reply_content(target, answer)}, "t", 1)
t, a = validate_inbox_reply(state, "s", item)
assert t == target and a == answer
item.payload["content"]["text"] = "forged"
try:
    validate_inbox_reply(state, "s", item)
except InteractionError as exc:
    assert exc.reason == "invalid_interaction_answer"
else:
    raise AssertionError("forged reply accepted")
raw = {"metadata": {"vv_artifact": {"private": "locator"}, "keep": [True, 1]}, "content": "ok"}
assert provider_result_identity(raw) == {"metadata": {"keep": [True, 1]}, "content": "ok"}
assert "vv_artifact" in raw["metadata"]
assert (
    compaction_request(
        InboxItem("c", "user", {"content": {"metadata": {"request_compaction": True}, "app_server_compaction": {}}})
    )
    == {}
)
assert all(
    n in {"vv_agent", "vv_agent.kernel"} or n.startswith("vv_agent.kernel.") for n in sys.modules if n.startswith("vv_agent")
)
"""
    result = subprocess.run(
        [sys.executable, "-I", *flags, "-B", "-c", script, str(Path(__file__).parents[1] / "src")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dependency_exports_are_actual_core_objects():
    from vv_agent import interaction
    from vv_agent.kernel import _interaction_values, _interactions, _summary_requests, _tool_result_checks
    from vv_agent.session import interactions, summary_requests, tool_results

    for name in (
        "HostInteractionRequest",
        "HostInteractionOutcome",
        "InteractionReplyTarget",
        "InteractionAnswerSpec",
        "InteractionAnswer",
        "AppServerUserInteraction",
        "derive_controller_command_id",
    ):
        assert getattr(interaction, name) is getattr(_interaction_values, name)
    for name in (
        "check_form",
        "user_interaction",
        "retained_interaction",
        "validate_reply",
        "reply_content",
        "validate_inbox_reply",
        "InteractionError",
    ):
        assert getattr(interactions, name) is getattr(_interactions, name)
    for name in ("compaction_request", "pending_compaction_requests", "summary_pending"):
        assert getattr(summary_requests, name) is getattr(_summary_requests, name)
    assert tool_results.provider_result_identity is _tool_result_checks.provider_result_identity
