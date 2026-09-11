#!/usr/bin/env python3
"""Run one vv-agent cycle at a time through Celery start/advance.

Start a worker first::

    cd vv-agent
    uv run celery -A examples.23_celery_backend worker -l info

Then run this module with a Redis broker configured by
``VV_AGENT_EXAMPLE_REDIS_URL``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from celery import Celery

from vv_agent import Agent, CheckpointConfig, RunConfig, Runner
from vv_agent.model import VvLlmModelProvider
from vv_agent.runtime.backends.celery import CeleryBackend, RuntimeRecipe, register_cycle_task
from vv_agent.runtime.backends.distributed import (
    CapabilityRef,
    DistributedCapabilities,
    DistributedCapabilityRegistry,
)
from vv_agent.runtime.stores.sqlite import SqliteCheckpointStore

REDIS_URL = os.getenv("VV_AGENT_EXAMPLE_REDIS_URL", "redis://localhost:6379/3")
SETTINGS_FILE = Path(os.getenv("VV_AGENT_LOCAL_SETTINGS", "local_settings.py"))
BACKEND_NAME = os.getenv("VV_AGENT_EXAMPLE_BACKEND", "moonshot")
MODEL = os.getenv("VV_AGENT_EXAMPLE_MODEL", "kimi-k3")
WORKSPACE = Path(os.getenv("VV_AGENT_EXAMPLE_WORKSPACE", "./workspace")).resolve()
CHECKPOINT_STORE = SqliteCheckpointStore(WORKSPACE / ".vv-agent-state" / "checkpoints.db")
CHECKPOINT_REF = CapabilityRef("checkpoint.example-23", "1")
CAPABILITIES = DistributedCapabilityRegistry()
CAPABILITIES.register("checkpoint_store", CHECKPOINT_REF, CHECKPOINT_STORE)

app = Celery("vv_agent_example", broker=REDIS_URL, backend=REDIS_URL)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
register_cycle_task(app, capability_registry=CAPABILITIES)


def build_backend() -> CeleryBackend:
    recipe = RuntimeRecipe(
        settings_file=str(SETTINGS_FILE),
        backend=BACKEND_NAME,
        model=MODEL,
        workspace=str(WORKSPACE),
        capabilities=DistributedCapabilities(checkpoint_store_ref=CHECKPOINT_REF),
    )
    return CeleryBackend(celery_app=app, runtime_recipe=recipe, capability_registry=CAPABILITIES)


def build_agent() -> Agent:
    return Agent(name="celery-demo", instructions="You are a helpful agent.", model=MODEL)


def build_run_config(backend: CeleryBackend, checkpoint_key: str) -> RunConfig:
    return RunConfig(
        model_provider=VvLlmModelProvider(SETTINGS_FILE, default_backend=BACKEND_NAME),
        execution_backend=backend,
        max_cycles=3,
        no_tool_policy="finish",
        checkpoint_config=CheckpointConfig(store=CHECKPOINT_STORE, key=checkpoint_key),
    )


def _continuation(_handle: Any, envelope: Any) -> Any:
    return advance_cycle.s(envelope.to_dict())


@app.task(name="examples.23_celery_backend.advance_cycle")
def advance_cycle(worker_result: dict[str, Any], envelope_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply one worker response and enqueue only the canonical next decision."""
    backend = build_backend()
    decision = backend.advance(
        previous_envelope=envelope_dict,
        outcome=worker_result,
        continuation=_continuation,
    )
    if decision.action == "finalize_required":
        result = Runner.finalize_distributed(
            build_agent(),
            "1+1 等于几?",
            decision=decision,
            run_config=build_run_config(backend, decision.handle.checkpoint_key),
        )
        print(f"[done] {result.status.value}: {result.final_output}", flush=True)
    elif decision.action == "terminal_replay":
        print(f"[replay] {decision.result.final_answer if decision.result else ''}", flush=True)
    elif decision.action == "wait":
        print(f"[wait] {decision.reason.value if decision.reason else 'pending'}", flush=True)
    return decision.to_dict()


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    backend = build_backend()
    checkpoint_key = f"celery_dist_{os.urandom(4).hex()}"
    handle = Runner.start_distributed(
        build_agent(),
        "1+1 等于几?",
        run_config=build_run_config(backend, checkpoint_key),
        continuation=_continuation,
    )
    print(f"[started] checkpoint={handle.checkpoint_key} run={handle.run_id}", flush=True)


if __name__ == "__main__":
    main()
