"""Interactive facade over the kernel inbox and record projections."""

from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

from vv_agent.agent import Agent
from vv_agent.interactive import AgentSession, AgentSessionRun, InteractiveAgentDefinition
from vv_agent.run_config import RunConfig
from vv_agent.types import AgentStatus, Message

from .kernel import drive
from .records import InboxItem, copy_json
from .result import project_result


class _Transcript:
    def __init__(self, kernel, session_id):
        self.kernel, self.session_id = kernel, session_id

    def get_items(self, limit=None):
        messages = self.kernel.messages(self.session_id)
        return messages if limit is None else messages[-limit:] if limit else []


class _KernelAgentSession(AgentSession):
    def __init__(self, *, client, agent, workspace, shared_state, session_id, seed_messages: list[Message]):
        self._client = client
        self._kernel = client._kernel
        sid = session_id or uuid4().hex[:12]
        self._kernel.create(
            sid, str(workspace), {"seed": {"messages": [m.to_dict() for m in seed_messages], "shared_state": shared_state or {}}}
        )
        definition = client._apply_startup_shell_defaults(agent) if isinstance(agent, InteractiveAgentDefinition) else None
        super().__init__(
            execute_run=client._execute,
            session_id=sid,
            agent_name="inline" if definition else agent.name,
            definition=definition,
            agent=None if definition else agent,
            workspace=workspace,
            shared_state=shared_state,
            session=cast(Any, _Transcript(self._kernel, sid)),
            approval_broker=client.options.approval_broker,
            parent_cancellation_token=client.options.cancellation_token,
            event_buffer_capacity=client.options.event_buffer_capacity,
        )
        state, _, _ = self._kernel.store.read_state(sid)
        self._closed = state.closed
        if state.turns:
            result = project_result(self._kernel.store, sid, next(reversed(state.turns)), runtime=self._runtime())
            self._shared_state = result.raw_result.shared_state
            self._latest_run = AgentSessionRun.from_run_result(result)

    def __getattribute__(self, name):
        if name in {"replace_messages", "replace_shared_state", "clear_queues", "session"}:
            raise AttributeError(f"Kernel AgentSession has no {name}")
        return super().__getattribute__(name)

    def __dir__(self):
        return [
            name
            for name in super().__dir__()
            if name not in {"replace_messages", "replace_shared_state", "clear_queues", "session"}
        ]

    @property
    def messages(self):
        return self._kernel.messages(self.session_id)

    @property
    def shared_state(self):
        state, rows, _ = self._kernel.store.read_state(self.session_id)
        if state.turns:
            return copy_json(
                project_result(self._kernel.store, self.session_id, next(reversed(state.turns))).raw_result.shared_state
            )
        return copy_json(rows[0].record.payload["attributes"].get("seed", {}).get("shared_state", {}))

    def _runtime(self):
        handle = self._active_run_handle
        if handle is None:
            handle = next((h for h in reversed(self._kernel.handles) if h.session_id == self.session_id), None)
        if handle is None:
            state, _, _ = self._kernel.store.read_state(self.session_id)
            definition = self.definition
            task = state.turns[next(reversed(state.turns))].start.task() if state.turns and definition else None
            agent = self.agent
            if agent is None:
                assert definition is not None
                agent = Agent(
                    self.agent_name,
                    task.prompt_bundle if task else definition.system_prompt or definition.description,
                    model=definition.model,
                    sub_agents=definition.sub_agents,
                )
            options = self._client.options
            config = RunConfig(
                model_provider=options.model_provider,
                workspace=self.workspace,
                tool_policy=options.tool_policy,
                approval_provider=options.approval_provider,
                approval_broker=self._approval_broker,
                approval_timeout_seconds=options.approval_timeout_seconds,
                tool_registry_factory=options.tool_registry_factory,
                hooks=options.runtime_hooks,
                memory_providers=[*options.memory_providers, *(definition.memory_providers if definition else [])],
                context_providers=[*options.context_providers, *(definition.context_providers if definition else [])],
                metadata={"session_id": self.session_id},
            )
            return self._kernel.runtime(agent, config, task)
        return handle.runtime

    def _persist_custom_run_delta(self, previous, run):
        del previous, run

    def _before_cycle_messages(self, cycle_index, _, __):
        return []

    def _interruption_messages(self):
        return []

    def steer(self, prompt):
        self._queue("steer", prompt)

    def follow_up(self, prompt):
        self._queue("follow_up", prompt)

    def _queue(self, kind, prompt):
        text = prompt.strip()
        if not text:
            raise ValueError(f"{kind} prompt cannot be empty")
        with self._lock:
            self._ensure_open_locked()
        state, _, _ = self._kernel.store.read_state(self.session_id)
        target = state.active_turn_id if kind == "steer" else None
        self._kernel.push(self.session_id, InboxItem(uuid4().hex, kind, {"content": text}, target))
        self._emit(f"session_{kind}_queued", prompt=text)

    def prompt(self, prompt, *, auto_follow_up=True):
        text = prompt.strip()
        if not text:
            raise ValueError("prompt cannot be empty")
        state, _, _ = self._kernel.store.read_state(self.session_id)
        if state.active_turn_id:
            self._kernel.answer(self.session_id, self._runtime(), text, uuid4().hex)
        run = self._run_once(text)
        while auto_follow_up and run.status == AgentStatus.COMPLETED and self._pending("follow_up"):
            run = self._run_once("")
        return run

    def continue_run(self, prompt=None):
        if prompt and prompt.strip():
            return self.prompt(prompt, auto_follow_up=False)
        state, _, _ = self._kernel.store.read_state(self.session_id)
        if not state.active_turn_id and not self._pending("follow_up"):
            raise ValueError("No queued prompt available. Provide prompt or call follow_up() first.")
        return self._run_once("")

    def approve(self, request_id, decision):
        self._kernel.approve(self.session_id, self._runtime(), request_id, decision, f"approval/{request_id}/answer")

    def archive(self, *, input_id="interactive/archive"):
        return self._kernel.control(self.session_id, "archive", input_id, runtime=self._runtime())

    def close(self, *, input_id="interactive/close"):
        state, _, _ = self._kernel.store.read_state(self.session_id)
        self._kernel.control(self.session_id, "close", input_id)
        changed = super().close()
        if not self._running:
            try:
                runtime = self._runtime()
            except RuntimeError:
                return changed
            drive(self._kernel.store, self.session_id, runtime=runtime, _one_turn=True)
        return changed and not state.closed

    def cancel(self):
        handle = self._active_run_handle
        return bool(handle and handle.cancel())

    def _pending(self, kind):
        state, _, _ = self._kernel.store.read_state(self.session_id)
        count = sum(i.item.kind == kind for i in self._kernel.store.peek_inbox(self.session_id))
        count += sum(
            r._payload["input"]["kind"] == kind and r._payload["disposition"] == "queued" and key not in state.admitted_inputs
            for key, r in state.applied_inputs.items()
        )
        return count

    def state(self):
        return replace(
            super().state(),
            messages=self._kernel.messages(self.session_id),
            shared_state=self.shared_state,
            pending_steering=self._pending("steer"),
            pending_follow_ups=self._pending("follow_up"),
        )
