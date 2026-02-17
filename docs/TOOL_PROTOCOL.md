# Tool Protocol (v-agent)

## Tool Naming

- Internal tools follow backend-style naming, e.g. `_task_finish`, `_ask_user`, `_read_file`.
- Tool schemas are single-source in `src/v_agent/constants/`.

## Tool Call Input

Each tool call contains:

- `id`: unique tool call id from LLM
- `name`: tool name
- `arguments`: JSON object

`ToolDispatcher` accepts dict or JSON string arguments and normalizes parse errors into structured `ERROR` results.

## Tool Result Model

`ToolExecutionResult` fields:

- `tool_call_id`
- `content` (JSON string payload)
- `status_code` (`ToolResultStatus`)
- `status` (legacy compatibility: `success`/`error`)
- `directive` (`continue`/`wait_user`/`finish`)
- `error_code` (optional)
- `metadata` (optional)
- `image_url` / `image_path` (optional)

## Status Codes

- `SUCCESS`: tool completed normally
- `ERROR`: tool failed with structured error payload
- `WAIT_RESPONSE`: tool asks runtime to pause for user response
- `RUNNING`: async/background work started and still in progress
- `BATCH_RUNNING`: batch async work in progress
- `PENDING_COMPRESS`: deferred memory-compression signal

## Runtime Rules

- `directive=wait_user` => task status `wait_user`
- `directive=finish` => task status `completed`
- `status_code=RUNNING` => remain in processing flow, wait for poll/check tool
- Image tools can set `image_url`/`image_path`; runtime appends image-loaded notifications to messages.

## Error Code Conventions

- Domain semantic errors should set stable `error_code` values (example: `todo_incomplete`).
- Dispatcher-level failures use protocol codes such as:
  - `invalid_arguments_json`
  - `invalid_arguments_payload`
  - `invalid_arguments_type`
  - `tool_not_found`
  - `tool_execution_failed`
