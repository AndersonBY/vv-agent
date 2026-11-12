"""Summary algorithms execute in core; the manager keeps settings, counting and notification."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_summary_core_import_and_canonical_producers_without_outer_sdk(tmp_path, site_flags):
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        root = Path(sys.argv[1])
        sys.path.insert(0, str(root / "src"))

        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel import _summary_compaction as core
        from vv_agent.kernel._canonical_json import canonical_json_bytes
        from vv_agent.kernel._message_values import Message

        assert Path(core.__file__) == root / "src/vv_agent/kernel/_summary_compaction.py"
        fixture = json.loads((root / "tests/fixtures/parity/memory_local.json").read_text())["summary_compaction"]
        for case in fixture["cases"]:
            if "token_estimator" not in case:
                continue
            inputs, estimator = case["input"], case["token_estimator"]
            messages = [Message.from_dict(value) for value in inputs["messages"]]
            plan = core.plan_summary(
                messages, keep_recent=inputs["keep_recent_messages"], drop_ratio=0,
                protected_start=None, language="zh-CN", event_limit=40,
            )
            assert type(plan) is core.SummaryPlan, case["name"]
            for tag, key in (("Previous Summary", "previous_summary"), ("Conversation Prefix", "conversation_prefix")):
                section = plan.prompt.split(f"<{tag}>\\n", 1)[1].split(f"\\n</{tag}>", 1)[0]
                assert section.encode() == canonical_json_bytes(case["expected_summary_input"][key]), case["name"]
            counts = []
            def count(value):
                counts.append(value)
                return estimator["input_messages"] if value == messages else estimator["candidate_messages"]
            steps = core.accept_summary(
                plan, inputs["summary_response"], protected_start=None,
                recovery_tool_available=inputs.get("recovery_tool_available", True),
                effective_context_window=10000, require_reduction=True,
            )
            request = next(steps)
            while True:
                value = count(request.messages)
                try:
                    request = steps.send(value)
                except StopIteration as done:
                    result = done.value
                    break
            assert result.accepted is case["expected"]["changed"], case["name"]
            assert canonical_json_bytes([message.to_dict() for message in result.messages]) == canonical_json_bytes(
                case["expected"]["messages"]
            ), case["name"]
            assert len(counts) == 2, case["name"]
            assert result.current_tokens == (estimator["candidate_messages"] if result.accepted else None)
            assert [message.to_dict() for message in messages] == inputs["messages"]

        prompt_case = fixture["prompt_cases"][0]
        case = next(case for case in fixture["cases"] if case["name"] == prompt_case["source_case"])
        messages = [Message.from_dict(value) for value in case["input"]["messages"]]
        for language, expected in prompt_case["expected_prompts"].items():
            plan = core.plan_summary(
                messages, keep_recent=case["input"]["keep_recent_messages"], drop_ratio=0,
                protected_start=None, language=language, event_limit=prompt_case["event_limit"],
            )
            assert plan.prompt.encode("utf-8") == expected.encode("utf-8")
        for case in fixture["invalid_block_cases"]:
            messages = [Message.from_dict(value) for value in case["messages"]]
            assert core.plan_summary(
                messages, keep_recent=1, drop_ratio=0, protected_start=None, language="en-US", event_limit=1,
            ) is None, case["name"]

        assert not hasattr(core, "MemoryManager")
        assert {name for name in sys.modules if name.startswith("vv_agent")} == {
            "vv_agent", "vv_agent.kernel", "vv_agent.kernel._summary_compaction",
            "vv_agent.kernel._canonical_json", "vv_agent.kernel._compaction_text",
            "vv_agent.kernel._context_messages", "vv_agent.kernel._message_values",
            "vv_agent.kernel._tool_values", "vv_agent.kernel._message_codec", "vv_agent.kernel._token_count",
        }
    """)
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_manager_uses_current_values_counting_seam_and_original_notification_count(monkeypatch):
    from vv_agent.kernel._context_messages import UnseenInputError
    from vv_agent.memory import MemoryManager, SessionMemory, SessionMemoryConfig
    from vv_agent.types import Message

    messages = [
        Message("system", "system"),
        Message("user", "history " * 100),
        Message("assistant", "work " * 100),
        Message("user", "protected"),
        Message("assistant", "tail"),
    ]
    calls, notifications = [], []

    def count(self, value):
        assert self is manager
        calls.append(value)
        return 1000 if value == messages else 77

    def on_compaction(self, *, current_tokens=None):
        assert self is manager.session_memory
        notifications.append(current_tokens)

    monkeypatch.setattr(MemoryManager, "_calculate_message_length", count)
    monkeypatch.setattr(SessionMemory, "on_compaction", on_compaction)
    manager = MemoryManager(keep_recent_messages=1, session_memory=SessionMemory(SessionMemoryConfig()))
    initial = manager.plan_summary(messages)
    assert initial is not None and initial.tail == messages[-1:]
    manager.keep_recent_messages = 2
    manager.language = "en-US"
    manager.summary_event_limit = 3
    manager.protected_start = 3
    plan = manager.plan_summary(messages)
    assert plan is not None and plan.tail == messages[3:]
    assert "Preserve up to 3 critical events" in plan.prompt
    assert calls == notifications == []

    manager.model_context_window = 78
    manager.reserved_output_tokens = 0
    candidate, accepted = manager.accept_summary(plan, '{"progress":["done"]}')
    assert accepted and candidate[-2:] == messages[3:]
    assert calls == [candidate, messages]
    assert notifications == [77]

    calls.clear()
    assert manager.accept_summary(plan, '{"progress":["done"]}', notify=False) == (candidate, True)
    assert calls == [candidate, messages] and notifications == [77]
    calls.clear()
    assert manager.accept_summary(plan, '{"progress":["done"]}', require_reduction=False) == (candidate, True)
    assert calls == [candidate] and notifications == [77, 77]

    calls.clear()
    manager.model_context_window = 76
    assert manager.accept_summary(plan, '{"progress":["done"]}') == (messages, False)
    assert calls == [candidate, messages] and notifications == [77, 77]
    calls.clear()
    manager.protected_start = 1
    with pytest.raises(UnseenInputError, match="protected suffix verbatim"):
        manager.accept_summary(plan, '{"progress":["done"]}')
    assert calls == [] and notifications == [77, 77]


def test_summary_plan_identity_and_full_history_planning():
    from vv_agent.kernel import _summary_compaction as core
    from vv_agent.memory import MemoryManager
    from vv_agent.memory.manager import SummaryPlan
    from vv_agent.session import compaction
    from vv_agent.types import Message

    assert SummaryPlan is compaction.SummaryPlan is core.SummaryPlan
    assert SummaryPlan.__module__ == core.__name__
    messages = [Message("system", "system"), Message("user", "work"), Message("assistant", "tail")]
    manager = MemoryManager(language="en-US")
    plan = manager.plan_full_summary(messages)
    assert type(plan) is SummaryPlan
    assert plan.prefix == messages[1:] and plan.tail == [] and plan.keep_recent == 0
    assert plan.messages == messages and plan.messages is not messages
