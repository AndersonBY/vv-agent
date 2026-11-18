from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from vv_agent.kernel._cycle_values import AFTER_CYCLE_CONTROL_SCHEMA as AFTER_CYCLE_CONTROL_SCHEMA
from vv_agent.kernel._cycle_values import AFTER_CYCLE_CONTROL_STATE_KEY as AFTER_CYCLE_CONTROL_STATE_KEY
from vv_agent.kernel._cycle_values import AfterCycleAction as AfterCycleAction
from vv_agent.kernel._cycle_values import AfterCycleDecision as AfterCycleDecision
from vv_agent.kernel._cycle_values import AfterCycleHookError as AfterCycleHookError
from vv_agent.kernel._cycle_values import AfterCycleSnapshot as AfterCycleSnapshot
from vv_agent.kernel._cycle_values import AfterCycleStop as AfterCycleStop
from vv_agent.kernel._cycle_values import NativeCycleOutcome as NativeCycleOutcome
from vv_agent.kernel._cycle_values import NativeCycleOutcomeKind as NativeCycleOutcomeKind
from vv_agent.kernel._cycle_values import persist_after_cycle_disallowed_tools as persist_after_cycle_disallowed_tools
from vv_agent.kernel._cycle_values import read_after_cycle_disallowed_tools as read_after_cycle_disallowed_tools


@runtime_checkable
class AfterCycleHook(Protocol):
    def after_cycle(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision | None: ...


@dataclass(slots=True)
class AfterCycleHookManager:
    hooks: list[AfterCycleHook] = field(default_factory=list)

    def has_hooks(self) -> bool:
        return bool(self.hooks)

    def apply(self, snapshot: AfterCycleSnapshot) -> AfterCycleDecision:
        steering_messages: list[str] = []
        disallow_tools: list[str] = []
        seen_tools: set[str] = set()
        for hook in self.hooks:
            try:
                decision = hook.after_cycle(snapshot)
            except Exception as exc:
                raise AfterCycleHookError(
                    f"after-cycle hook failed: {exc}",
                    code="after_cycle_hook_failed",
                ) from exc
            if decision is None:
                continue
            if not isinstance(decision, AfterCycleDecision):
                raise AfterCycleHookError(
                    "after-cycle hook returned an invalid decision",
                    code="after_cycle_decision_invalid",
                )
            for tool_name in decision.disallow_tools:
                if tool_name not in seen_tools:
                    seen_tools.add(tool_name)
                    disallow_tools.append(tool_name)
            if decision.action is AfterCycleAction.STOP_NON_SUCCESS:
                return decision
            if decision.action is AfterCycleAction.STEER:
                steering_messages.extend(decision.steering_messages)
        if steering_messages:
            try:
                return AfterCycleDecision.steer(
                    steering_messages,
                    disallow_tools=disallow_tools,
                )
            except (TypeError, ValueError) as exc:
                raise AfterCycleHookError(
                    f"composed after-cycle decision is invalid: {exc}",
                    code="after_cycle_decision_invalid",
                ) from exc
        try:
            return AfterCycleDecision.continue_run(disallow_tools=disallow_tools)
        except (TypeError, ValueError) as exc:
            raise AfterCycleHookError(
                f"composed after-cycle decision is invalid: {exc}",
                code="after_cycle_decision_invalid",
            ) from exc
