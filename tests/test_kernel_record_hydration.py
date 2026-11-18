"""Pure kernel hydration stays available without importing the SDK session layer."""

from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = r"""
import importlib
import importlib.abc
import json
import sys
from pathlib import Path

source, root, purelib = sys.argv[1:]
sys.path[:0] = [source, root, purelib]

class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vv_agent.session" or fullname.startswith("vv_agent.session."):
            raise AssertionError(f"forbidden SDK import: {fullname}")
        return None

sys.meta_path.insert(0, CoreOnly())
hydration_module = importlib.import_module("vv_agent.kernel._boundary_hydration")
from vv_agent.kernel._canonical_json import canonical_json_bytes
from vv_agent.kernel._content import ContentLabels, ContentRef
from vv_agent.kernel._records import InboxItem

content_body = canonical_json_bytes({"text": "hello"}, "input")
actor_body = canonical_json_bytes({"role": "operator"}, "actor")

def ref(content_id, body):
    return ContentRef(content_id, "a" * 64, len(body), "key/test", ContentLabels())

content_ref = ref("content/input", content_body)
actor_ref = ref("content/actor", actor_body)

class Reader:
    binding = ("tenant", "session")

    def __init__(self):
        self.calls = []

    def read(self, tenant_id, session_id, owner, slot, requested):
        self.calls.append((tenant_id, session_id, owner, slot, requested.content_id))
        if slot == "input.content":
            return content_body
        if slot == "input.actor.attributes":
            return actor_body
        raise AssertionError(slot)

reader = Reader()
raw = InboxItem(
    "input-1",
    "user",
    {"content": content_ref.to_dict()},
    actor={
        "tenant_id": "tenant",
        "subject_id": "subject",
        "subject_kind": "user",
        "attributes": actor_ref.to_dict(),
    },
)
hydration = hydration_module.SessionHydration(reader, tenant_id="tenant", session_id="session")
projected = hydration_module.hydrate_session_input(raw, hydration)
verified = hydration_module.hydrate_session_input(raw, hydration, include_actor_attributes=True)

assert projected.payload["content"] == {"text": "hello"}
assert projected.actor["attributes"] == actor_ref.to_dict()
assert verified.payload["content"] == {"text": "hello"}
assert verified.actor["attributes"] == {"role": "operator"}
assert [call[3] for call in reader.calls] == [
    "input.content",
    "input.content",
    "input.actor.attributes",
]
assert not any(name == "vv_agent.session" or name.startswith("vv_agent.session.") for name in sys.modules)
assert callable(hydration_module.hydrate_and_fold_records)
assert hydration_module.RetainedInboxItem.__name__ == "RetainedInboxItem"
print(json.dumps({"calls": reader.calls, "session_imported": False, "verified": True}))
"""


def test_pure_core_hydration_uses_input_owner_without_session_import() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            PROBE,
            str(ROOT / "src"),
            str(ROOT),
            sysconfig.get_paths()["purelib"],
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["verified"] is True
    assert evidence["session_imported"] is False
    assert evidence["calls"] == [
        ["tenant", "session", "input:input-1", "input.content", "content/input"],
        ["tenant", "session", "input:input-1", "input.content", "content/input"],
        ["tenant", "session", "input:input-1", "input.actor.attributes", "content/actor"],
    ]
