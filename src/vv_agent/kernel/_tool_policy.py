"""Tool policies and the shared context-policy metadata composition."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ._cycle_values import read_after_cycle_disallowed_tools
from ._tool_metadata import ToolSideEffect, normalize_denied_side_effects, normalize_metadata_labels

ApprovalPolicy = Literal["default", "always", "never", "on_request"]


_APPROVAL_POLICIES = frozenset({"default", "always", "never", "on_request"})


CanUseTool = Callable[[str, dict[str, Any]], bool]


@dataclass(slots=True)
class ToolPolicy:
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    approval: ApprovalPolicy = "default"
    can_use_tool: CanUseTool | None = None
    denied_side_effects: Sequence[ToolSideEffect | str] = field(default_factory=list)
    denied_capability_tags: list[str] = field(default_factory=list)
    deny_terminal_tools: bool = False
    denied_cost_dimensions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.approval not in _APPROVAL_POLICIES:
            supported = ", ".join(sorted(_APPROVAL_POLICIES))
            raise ValueError(f"approval must be one of: {supported}")
        self.denied_side_effects = normalize_denied_side_effects(self.denied_side_effects)
        self.denied_capability_tags = normalize_metadata_labels(
            self.denied_capability_tags,
            field_name="denied_capability_tags",
        )
        if not isinstance(self.deny_terminal_tools, bool):
            raise TypeError("deny_terminal_tools must be a boolean")
        self.denied_cost_dimensions = normalize_metadata_labels(
            self.denied_cost_dimensions,
            field_name="denied_cost_dimensions",
        )


def merge_tool_policies(
    agent_policy: ToolPolicy | None,
    run_policy: ToolPolicy | None,
) -> ToolPolicy | None:
    return merge_tool_policy_layers(agent_policy, None, run_policy)


def merge_tool_policy_layers(
    agent_policy: ToolPolicy | None,
    runner_policy: ToolPolicy | None,
    run_policy: ToolPolicy | None,
) -> ToolPolicy | None:
    if agent_policy is None and runner_policy is None and run_policy is None:
        return None

    agent = agent_policy or ToolPolicy()
    runner = runner_policy or ToolPolicy()
    run = run_policy or ToolPolicy()
    allowed_tools = next(
        (policy.allowed_tools for policy in (run, runner, agent) if policy.allowed_tools is not None),
        None,
    )
    disallowed_tools = list(dict.fromkeys([*agent.disallowed_tools, *runner.disallowed_tools, *run.disallowed_tools]))
    denied_side_effects = normalize_denied_side_effects(
        [*agent.denied_side_effects, *runner.denied_side_effects, *run.denied_side_effects]
    )
    denied_capability_tags = normalize_metadata_labels(
        [
            *agent.denied_capability_tags,
            *runner.denied_capability_tags,
            *run.denied_capability_tags,
        ],
        field_name="denied_capability_tags",
    )
    denied_cost_dimensions = normalize_metadata_labels(
        [
            *agent.denied_cost_dimensions,
            *runner.denied_cost_dimensions,
            *run.denied_cost_dimensions,
        ],
        field_name="denied_cost_dimensions",
    )

    approval = next(
        (policy.approval for policy in (run, agent, runner) if policy.approval != "default"),
        "default",
    )
    can_use_tool = _merge_can_use_tool(
        _merge_can_use_tool(agent.can_use_tool, runner.can_use_tool),
        run.can_use_tool,
    )
    return ToolPolicy(
        allowed_tools=list(allowed_tools) if allowed_tools is not None else None,
        disallowed_tools=disallowed_tools,
        approval=approval,
        can_use_tool=can_use_tool,
        denied_side_effects=denied_side_effects,
        denied_capability_tags=denied_capability_tags,
        deny_terminal_tools=(agent.deny_terminal_tools or runner.deny_terminal_tools or run.deny_terminal_tools),
        denied_cost_dimensions=denied_cost_dimensions,
    )


def _merge_can_use_tool(
    agent_predicate: CanUseTool | None,
    run_predicate: CanUseTool | None,
) -> CanUseTool | None:
    if agent_predicate is None:
        return run_predicate
    if run_predicate is None:
        return agent_predicate

    def can_use_tool(tool_name: str, arguments: dict[str, Any]) -> bool:
        return bool(agent_predicate(tool_name, dict(arguments))) and bool(run_predicate(tool_name, dict(arguments)))

    return can_use_tool


_TASK_TOOL_POLICY_METADATA_KEYS = (
    "_vv_agent_allowed_tools",
    "_vv_agent_disallowed_tools",
    "_vv_agent_denied_side_effects",
    "_vv_agent_denied_capability_tags",
    "_vv_agent_deny_terminal_tools",
    "_vv_agent_denied_cost_dimensions",
)


def _apply_tool_policy_metadata(
    metadata: dict[str, Any],
    policy: ToolPolicy | None,
) -> None:
    for key in _TASK_TOOL_POLICY_METADATA_KEYS:
        metadata.pop(key, None)
    if policy is None:
        return
    if policy.allowed_tools is not None:
        metadata["_vv_agent_allowed_tools"] = list(policy.allowed_tools)
    if policy.disallowed_tools:
        metadata["_vv_agent_disallowed_tools"] = list(policy.disallowed_tools)
    if policy.denied_side_effects:
        metadata["_vv_agent_denied_side_effects"] = [ToolSideEffect(item).value for item in policy.denied_side_effects]
    if policy.denied_capability_tags:
        metadata["_vv_agent_denied_capability_tags"] = list(policy.denied_capability_tags)
    if policy.deny_terminal_tools:
        metadata["_vv_agent_deny_terminal_tools"] = True
    if policy.denied_cost_dimensions:
        metadata["_vv_agent_denied_cost_dimensions"] = list(policy.denied_cost_dimensions)


def apply_tool_context_policy(
    metadata: dict[str, Any],
    *,
    agent_policy: ToolPolicy | None,
    run_policy: ToolPolicy | None,
    shared_state: dict[str, Any] | None,
) -> None:
    frozen = ToolPolicy(
        allowed_tools=metadata.get("_vv_agent_allowed_tools"),
        disallowed_tools=metadata.get("_vv_agent_disallowed_tools", []),
        denied_side_effects=metadata.get("_vv_agent_denied_side_effects", []),
        denied_capability_tags=metadata.get("_vv_agent_denied_capability_tags", []),
        denied_cost_dimensions=metadata.get("_vv_agent_denied_cost_dimensions", []),
        deny_terminal_tools=metadata.get("_vv_agent_deny_terminal_tools", False),
        approval=metadata.get("_vv_agent_tool_policy_approval", "default"),
    )
    current = merge_tool_policies(agent_policy, run_policy)
    policy = merge_tool_policies(frozen, current)
    assert policy is not None
    if frozen.allowed_tools is not None and policy.allowed_tools is not None:
        policy.allowed_tools = [n for n in frozen.allowed_tools if n in policy.allowed_tools]
    if frozen.approval == "always":
        policy.approval = "always"
    policy.disallowed_tools = list(
        dict.fromkeys([*policy.disallowed_tools, *read_after_cycle_disallowed_tools(shared_state or {})])
    )
    _apply_tool_policy_metadata(metadata, policy)
    metadata["_vv_agent_tool_policy_approval"] = policy.approval
    metadata["_vv_agent_tool_policy_can_use_tool"] = policy.can_use_tool
