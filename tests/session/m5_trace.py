"""Opt-in call tracing for M5, inherited by spawned workers via sitecustomize."""

import atexit
import json
import os
import sys
import threading
from collections import Counter
from functools import cache
from pathlib import Path
from uuid import uuid4

VV_ROOT = str(Path(__file__).resolve().parents[2] / "src" / "vv_agent")
OUTPUT = Path(os.environ["KERNEL_M5_TRACE_DIR"])
DESTINATION = OUTPUT / f"calls-{os.getpid()}-{uuid4().hex[:12]}.json"
CALLS = Counter()
ORIGINS = {}
LOCK = threading.RLock()


@cache
def prototype(path):
    return (
        path.startswith(VV_ROOT + "/session/")
        or "/ai_agents/services/agent/session_kernel/" in path
        or path.endswith("/ai_agents/tasks/session_kernel_tasks.py")
    )


@cache
def source(path):
    return path.startswith(VV_ROOT + "/") or "/ai_agents/" in path


def flush():
    with LOCK:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "calls": [
                {
                    "file": path,
                    "function": name,
                    "line": line,
                    "count": count,
                    "value_type_origins": dict(ORIGINS.get((path, name, line), {})),
                }
                for (path, name, line), count in sorted(CALLS.items())
            ],
        }
        temporary = DESTINATION.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(DESTINATION)


def trace(frame, event, arg):
    if event != "call":
        return
    code = frame.f_code
    path = code.co_filename
    # Module/class body execution is import-time definition, not a runtime function call.
    if source(path) and code.co_flags & 2:
        ancestor = frame
        while ancestor is not None and not prototype(ancestor.f_code.co_filename):
            ancestor = ancestor.f_back
        if ancestor is not None:
            with LOCK:
                key = (path, code.co_qualname, code.co_firstlineno)
                CALLS[key] += 1
                if path.endswith("/interaction.py"):
                    caller = frame
                    while caller is not None:
                        if caller.f_code.co_qualname.startswith("HostInteractionRequest."):
                            ORIGINS.setdefault(key, Counter())[caller.f_code.co_qualname] += 1
                            break
                        caller = caller.f_back
    # Save before a barrier can block and the worker is SIGKILLed (atexit won't run).
    if code.co_name == "hook" and ("/tests/session/" in path or path.endswith("/ai_agents/tests/session_kernel_support.py")):
        flush()


if os.environ.get("KERNEL_M5_TRACE_DIR"):
    sys.setprofile(trace)
    threading.setprofile(trace)
    atexit.register(flush)
