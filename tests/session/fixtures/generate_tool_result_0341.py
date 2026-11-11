import importlib.metadata
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from tests.session.test_recovery_matrix import effect_tool, runtime, start
from tests.session.test_tool_result_artifacts import LongProvider
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.types import LLMResponse, ToolCall
from vv_agent.workspace import LocalWorkspaceBackend

assert importlib.metadata.version("vv-agent") == "0.34.1"
with tempfile.TemporaryDirectory(prefix="vva-fix2-old-") as root:
    root = Path(root)
    database = root / "session.sqlite"
    with SQLiteStore.standalone(str(database)) as store:
        store.install_schema()
        provider = LongProvider(database)
        provider.install()
        start(store)
        backend = LocalWorkspaceBackend(root / "workspace")
        rt = runtime(
            database,
            [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
            [effect_tool()],
            providers={"effect": provider},
        )
        rt.config = replace(rt.config, workspace_backend=backend)
        rt.workspace_backend = backend
        drive(store, "s", runtime=rt)
        _, records, _ = read_state(store, "s")
        completion = next(
            r.record for r in records if r.record.kind == "op_completed" and r.record.payload["result"].get("tool_call_id") == "a"
        )
        artifact = completion.payload["result"]["artifact"]
        callback = provider.callback()
        assert backend.read_text(artifact["path"]) == callback.payload["result"]["content"]
        fixture = {
            "producer": importlib.metadata.version("vv-agent"),
            "records": [r.record.to_dict() for r in records],
            "callback": json.loads(callback.encode()),
        }
        Path(sys.argv[1]).write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
        print(
            "LEGACY_PRODUCER",
            {
                "version": fixture["producer"],
                "source": sys.modules["vv_agent"].__file__,
                "records": len(records),
                "artifact": artifact,
            },
        )
