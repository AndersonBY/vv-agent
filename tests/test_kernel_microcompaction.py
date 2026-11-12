"""The complete microcompaction planner uses only core values and supplied estimates."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("site_flags", [[], ["-S"]], ids=["normal", "no-site"])
def test_microcompaction_import_and_callbacks_without_outer_sdk(tmp_path, site_flags):
    source = Path(__file__).resolve().parents[1] / "src"
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from hashlib import sha256
        from pathlib import Path

        sys.path.insert(0, sys.argv[1])

        class CoreOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"vv_agent", "vv_agent.kernel"} or fullname.startswith("vv_agent.kernel."):
                    return None
                assert fullname.split(".")[0] in sys.stdlib_module_names, fullname

        sys.meta_path.insert(0, CoreOnly())
        from vv_agent.kernel import _microcompaction as core
        from vv_agent.kernel._compaction_text import COMPACT_MARKER_OPENING
        from vv_agent.kernel._message_values import Message, ToolArtifactRef
        from vv_agent.kernel._microcompaction_policy import MicrocompactionPolicy
        from vv_agent.kernel._tool_metadata import ToolResultRetention

        assert Path(core.__file__) == Path(sys.argv[1]) / "vv_agent/kernel/_microcompaction.py"
        body = "汉🙂" * 400
        artifact = ToolArtifactRef(
            path=".vv-agent/artifacts/run/existing.txt", media_type="text/plain", encoding="utf-8",
            size_bytes=len(body.encode("utf-8")), sha256=sha256(body.encode("utf-8")).hexdigest(),
        )
        def turn(call_id, content, **kwargs):
            return [
                Message("assistant", "", tool_calls=[{"id": call_id, "function": {"name": call_id}}]),
                Message("tool", content, tool_call_id=call_id, **kwargs),
            ]

        messages = [
            *turn("existing", body, artifact_ref=artifact,
                  metadata={core.EXCERPT_METADATA_KEY: "片段🙂TAIL", "keep": [1]}),
            *turn("new", "β" * 900),
            *turn("preserved", "p" * 900),
            *turn("recovery", "r" * 900 + '\\n{"vv_agent_recovery":{}}'),
            *turn("compacted", COMPACT_MARKER_OPENING + "c" * 900),
            *turn("recent", "recent" * 900),
            Message("assistant", "summary", name="memory_summary"),
        ]
        path_calls, estimates = [], []
        def path_for(call_id):
            path_calls.append(call_id)
            return f".vv-agent/artifacts/run/{call_id}.txt"
        def estimate(message):
            assert type(message) is Message
            estimates.append(message)
            return len(message.content)

        plan = core.plan_microcompact(
            messages, current_cycle=9999, current_tokens=5000, target_tokens=3000,
            policy=MicrocompactionPolicy(keep_recent_cycles=1, min_result_chars=500),
            result_retentions={"preserved": ToolResultRetention.PRESERVE},
            artifact_path_estimate_for=path_for, estimate_message_tokens=estimate,
            excerpt_head_chars=2, excerpt_tail_chars=2,
        )
        assert type(plan) is core.MicrocompactPlan
        assert [candidate.message_index for candidate in plan.candidates] == [1, 3]
        assert [candidate.tool_call_id for candidate in plan.candidates] == ["existing", "new"]
        assert path_calls == ["new"]
        assert estimates[0] is messages[1] and estimates[2] is messages[3]
        assert len(estimates) == 4
        assert plan.candidates[0].existing_artifact is artifact
        assert plan.candidate_count == 2
        assert plan.estimated_reclaimable_tokens == sum(
            len(estimates[i].content) - len(estimates[i + 1].content) for i in (0, 2)
        )
        marker = (
            "<Tool Result Compact>\\ntool_name: existing\\n"
            "artifact_path: .vv-agent/artifacts/run/existing.txt\\n"
            "retrieval_hint: use read_file on artifact_path if needed\\n"
            "excerpt:\\n片段\\n...<snip>...\\nIL\\n</Tool Result Compact>"
        )
        assert estimates[1].content.encode("utf-8") == marker.encode("utf-8")
        replacement = core.replace_with_compacted_marker(
            messages[1], plan.candidates[0], artifact=artifact, marker=marker,
        )
        assert type(replacement) is Message and replacement is not messages[1]
        assert replacement.artifact_ref is artifact and replacement.metadata == {"keep": [1]}
        assert replacement.tool_call_id == messages[1].tool_call_id
        assert messages[1].content == body
        assert messages[1].metadata[core.EXCERPT_METADATA_KEY] == "片段🙂TAIL"
        assert {name for name in sys.modules if name.startswith("vv_agent")} == {
            "vv_agent", "vv_agent.kernel", "vv_agent.kernel._microcompaction",
            "vv_agent.kernel._compaction_text", "vv_agent.kernel._message_values",
            "vv_agent.kernel._canonical_json", "vv_agent.kernel._microcompaction_policy",
            "vv_agent.kernel._tool_metadata", "vv_agent.kernel._summary_compaction",
            "vv_agent.kernel._compaction_threshold", "vv_agent.kernel._context_messages",
            "vv_agent.kernel._tool_values",
        }
    """)
    result = subprocess.run(
        [sys.executable, "-I", *site_flags, "-B", "-c", script, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_microcompaction_keeps_public_and_active_caller_identity():
    from vv_agent import memory
    from vv_agent.kernel import _compaction_text as text
    from vv_agent.kernel import _microcompaction as core
    from vv_agent.kernel import _reducer as reducer
    from vv_agent.kernel._microcompaction_policy import MicrocompactionPolicy
    from vv_agent.kernel._tool_metadata import ToolResultRetention
    from vv_agent.memory import manager, microcompact
    from vv_agent.session import kernel

    for name in microcompact.__all__:
        owner = text if hasattr(text, name) else core
        assert getattr(microcompact, name) is getattr(owner, name), name
    assert memory.MicrocompactPlan is core.MicrocompactPlan
    assert memory.plan_microcompact is core.plan_microcompact
    for name in ("MicrocompactCandidate", "MicrocompactPlan", "plan_memory_microcompact", "replace_with_compacted_marker"):
        value = getattr(core, name)
        assert value.__module__ == core.__name__, name
        assert getattr(manager, name) is value, name
    assert manager.EXCERPT_METADATA_KEY is reducer.EXCERPT_METADATA_KEY is core.EXCERPT_METADATA_KEY
    assert reducer.replace_with_compacted_marker is core.replace_with_compacted_marker
    assert kernel.is_microcompacted_tool_content is text.is_microcompacted_tool_content
    assert core.MicrocompactionPolicy is manager.MicrocompactionPolicy is MicrocompactionPolicy
    assert core.ToolResultRetention is manager.ToolResultRetention is ToolResultRetention
    assert not any(name.startswith("_") and not name.startswith("__") for name in vars(microcompact))
