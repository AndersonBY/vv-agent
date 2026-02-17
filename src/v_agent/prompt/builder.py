from __future__ import annotations

from datetime import UTC, datetime

from v_agent.prompt.templates import (
    ASK_USER_PROMPT,
    COMPUTER_AGENT_ENV_PROMPT,
    CURRENT_TIME_PROMPT,
    TASK_FINISH_PROMPT,
    TODO_PROMPT,
    TOOL_PRIORITY_PROMPT,
    render_workspace_tools,
)


def build_system_prompt(
    original_system_prompt: str,
    *,
    language: str = "en-US",
    allow_interruption: bool = True,
    use_workspace: bool = True,
    enable_todo_management: bool = True,
    agent_type: str | None = None,
    current_time_utc: datetime | None = None,
) -> str:
    prompt_sections: list[str] = [f"<Agent Definition>\n{original_system_prompt}\n</Agent Definition>"]

    if agent_type == "computer":
        environment_text = COMPUTER_AGENT_ENV_PROMPT.get(language, COMPUTER_AGENT_ENV_PROMPT["en-US"])
        prompt_sections.append(f"<Environment>\n{environment_text}\n</Environment>")

    tools_lines: list[str] = []
    if allow_interruption:
        tools_lines.append(ASK_USER_PROMPT.get(language, ASK_USER_PROMPT["en-US"]))
    if use_workspace:
        tools_lines.append(render_workspace_tools(language))
        tools_lines.append(TOOL_PRIORITY_PROMPT.get(language, TOOL_PRIORITY_PROMPT["en-US"]))
    if enable_todo_management:
        tools_lines.append(TODO_PROMPT.get(language, TODO_PROMPT["en-US"]))
    tools_lines.append(TASK_FINISH_PROMPT.get(language, TASK_FINISH_PROMPT["en-US"]))
    prompt_sections.append(f"<Tools>\n{'\n\n'.join(tools_lines)}\n</Tools>")

    now = current_time_utc or datetime.now(tz=UTC)
    now_text = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_header = CURRENT_TIME_PROMPT.get(language, CURRENT_TIME_PROMPT["en-US"])
    prompt_sections.append(f"<Current Time>\n{time_header}\n{now_text}\n</Current Time>")

    return "\n\n".join(prompt_sections)
