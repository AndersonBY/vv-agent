from __future__ import annotations

from copy import deepcopy
from typing import Any

from vv_agent.constants.tool_names import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    COMPRESS_MEMORY_TOOL_NAME,
    CREATE_SUB_TASK_TOOL_NAME,
    EDIT_FILE_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    FIND_FILES_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    SEARCH_FILES_TOOL_NAME,
    SUB_TASK_STATUS_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
)

ToolSchema = dict[str, Any]

WORKSPACE_TOOLS = [
    FIND_FILES_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    READ_FILE_TOOL_NAME,
    WRITE_FILE_TOOL_NAME,
    EDIT_FILE_TOOL_NAME,
    SEARCH_FILES_TOOL_NAME,
    COMPRESS_MEMORY_TOOL_NAME,
    TODO_WRITE_TOOL_NAME,
]

WORKSPACE_TOOLS_SCHEMAS: dict[str, ToolSchema] = {
    READ_FILE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_FILE_TOOL_NAME,
            "description": """Read file contents from workspace.

When to use:
- Inspect source files, configs, logs, docs, generated artifacts, or exact snippets without shelling out to cat/head/tail.
- Read large files in chunks after using `file_info` or after a truncated response suggests a narrower line range.
- Use `show_line_numbers=true` when you need to quote lines, plan precise edits, or coordinate with `edit_file`.

Supported behavior:
- Reads plain UTF-8 text files and returns a content slice.
- Uses 1-based line numbers for `start_line` and `end_line`.
- Can prepend line numbers with `show_line_numbers=true`.
- Enforces read limits per request: max 2000 lines or 50000 characters.
- Large reads return file info payload instead of full content.

Guidance:
- Prefer this tool instead of shell commands like cat/head/tail.
- For large files, read in chunks by line range.
- By default, paths are workspace-relative.
- If runtime metadata enables outside-workspace access, absolute local paths are allowed.

Returns:
- A UTF-8 text slice with path metadata, requested line range, actual returned range, and optional line numbers.
- If the request exceeds safe limits, a file-info style payload with file statistics and suggested smaller ranges instead of \
flooding the LLM context.

Safety and limits:
- Uses 1-based inclusive line numbers for `start_line` and `end_line`.
- Enforces max 2000 lines or 50000 characters per request.
- Prefer `file_info` before reading unknown large or binary-looking paths.
- Paths are workspace-relative by default; absolute local paths require explicit outside-workspace runtime permission.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Target file path (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Optional starting line number (1-based). Use with `end_line` to read a chunk "
                            "from a large file instead of loading the whole file."
                        ),
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Optional ending line number (1-based, inclusive). Pair with `start_line` "
                            "to keep large-file reads bounded."
                        ),
                    },
                    "show_line_numbers": {
                        "type": "boolean",
                        "description": (
                            "When true, prefixes each output line with its source line number. Enable when "
                            "you need to quote precise lines, plan edits, or compare snippets."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    WRITE_FILE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": WRITE_FILE_TOOL_NAME,
            "description": """Write content to a file in workspace.

MODES:
- Overwrite (default): Replaces entire file content.
- Append: Adds to existing content (`append=true`).

WARNING:
- By default, this OVERWRITES the entire file.
- Use `append=true` to add content instead.

PARAMETERS:
- `path` (required): Workspace-relative path by default. Absolute path is allowed when outside-workspace access is enabled.
- `content` (required): Content to write.
- `append` (optional): Set true to append instead of overwrite.
- `leading_newline`/`trailing_newline` (optional): Add newlines when appending.

When to use:
- Create a new file, replace an entire generated artifact, or append a clearly bounded section to an existing file.
- Use `append=true` only when preserving all existing content is intentional and the appended block boundary is clear.
- Prefer `edit_file` for small or surgical edits to existing files.

Do not use this for surgical edits to existing source files; prefer `edit_file` when current read context is known through \
`read_file`, a full `write_file`, or a previous successful `edit_file`.
Appending to an unknown existing file does not create a full edit baseline; call `read_file` before editing after that case.

Returns:
- Structured write metadata including normalized path, append mode, character count, and newline flags.
- Errors when the path escapes the workspace or the backend refuses the write.

Safety and behavior:
- Overwrite is the default and replaces the whole file.
- This can create parent directories when the workspace backend supports it.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Target file path (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "The complete file body for overwrite mode, or the exact block to append when "
                            "`append=true`. When overwriting, preserve existing content yourself; use "
                            "`append=true` only when the existing file should remain intact."
                        ),
                    },
                    "append": {
                        "type": "boolean",
                        "description": (
                            "Set true to append instead of overwrite. Default is false (overwrite). "
                            "Use append only when existing content must be preserved."
                        ),
                    },
                    "leading_newline": {
                        "type": "boolean",
                        "description": (
                            "Add a leading newline when appending. Default is false. Use as a separator "
                            "from existing content when needed."
                        ),
                    },
                    "trailing_newline": {
                        "type": "boolean",
                        "description": (
                            "Add a trailing newline when appending. Default is false. Use to preserve a line "
                            "boundary before the next append or shell read."
                        ),
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    FIND_FILES_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": FIND_FILES_TOOL_NAME,
            "description": """Find files in workspace with optional path and glob filtering.

Large results are truncated, and common dependency/cache directories
(like node_modules/.venv) are summarized by default when listing from workspace root.

When to use:
- Discover repository structure, find candidate files before reading them, or inspect generated output locations.
- Use `path` to narrow the search root and `glob` to narrow file names before broad scans.
- Set `include_hidden=true` or `include_ignored=true` only when the task specifically needs those normally skipped paths.
- Set `include_sensitive=true` only when the task explicitly needs files that look like secrets, credentials, keys, tokens, \
or private config.

Narrow first:
- Common dependency/cache directories such as node_modules, .venv, target, and build outputs are summarized from \
workspace-root listings by default.
- Large results are truncated; use the returned `truncated`, `returned_count`, `max_results`, `remaining_count`, and \
`ignored_roots` fields to choose a smaller follow-up query.
- When a backend scan limit is reached, the response can include `count_is_estimate=true`.
- Use `offset` for pagination and `sort` to choose `modified_desc` or `path_asc`.

Returns:
- A structured list of normalized file paths plus counts, truncation metadata, ignored root summaries, and scan-limit hints.
- Errors for invalid paths or paths outside the permitted workspace.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional search root path. Use workspace-relative path by default; "
                            "absolute path is allowed when outside-workspace access is enabled. "
                            "Default '.'."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": (
                            "Optional glob filter such as `**/*.rs` or `src/**/*.md`. Use it to narrow by "
                            "filename, directory, or extensions before listing broad trees. Default **/*."
                        ),
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": (
                            "Whether hidden files and dotfiles are included. Default false; set true only "
                            "when the task explicitly needs paths such as .env.example, .github, or other "
                            "hidden project files."
                        ),
                    },
                    "include_ignored": {
                        "type": "boolean",
                        "description": (
                            "When listing workspace root, include files under common "
                            "dependency/cache directories. Default false; set true only when explicitly "
                            "inspecting generated, dependency, cache, or build-output paths."
                        ),
                    },
                    "include_sensitive": {
                        "type": "boolean",
                        "description": (
                            "Include files whose paths look like secrets, credentials, keys, tokens, "
                            "or private config. Default false."
                        ),
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["modified_desc", "path_asc"],
                        "description": (
                            "Sort order. `modified_desc` uses local file modification time when available; "
                            "non-local backends may fall back to `path_asc`. Default modified_desc."
                        ),
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of matching file paths to skip before returning results. Default 0.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Maximum number of file paths returned in one call. "
                            "Default 100; larger values are capped. If truncated, use returned counts "
                            "to run a narrower follow-up query."
                        ),
                    },
                    "scan_limit": {
                        "type": "integer",
                        "description": (
                            "Maximum files scanned before stopping early to keep listing fast. "
                            "If reached, response includes `count_is_estimate=true`."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    FILE_INFO_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": FILE_INFO_TOOL_NAME,
            "description": """Read file metadata in workspace, including size, modified time and type.

Inspect file metadata in workspace without loading full contents.

When to use:
- Use before reading large or binary files.
- Use before deciding read ranges, or before editing a path whose size/type is unknown.
- Check whether a path is a file or directory and whether it has a suffix that suggests text, image, archive, or binary content.
- Estimate whether `read_file`, `read_image`, or a narrower grep/search is the right next tool.

Returns:
- Normalized path, file/dir flags, byte size, modified time, suffix, and line count when it can be determined safely.
- Structured errors for missing paths or paths outside the permitted workspace.

Safety:
- This is a metadata probe; it should be preferred over reading a whole unknown file just to decide what to do next.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Target file path (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    SEARCH_FILES_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": SEARCH_FILES_TOOL_NAME,
            "description": """Search workspace file contents with regex or literal text.

When to use:
- Find symbols, text, config keys, error strings, TODOs, or call sites before deciding which files to read or edit.
- Prefer this tool over ad-hoc shell grep for direct content search.
- Narrow broad searches with `path`, `glob`, or `type` so results stay useful and fast.

OUTPUT MODES:
- `files_with_matches` (default): show only matching file paths.
- `content`: show matching lines with optional context and line numbers.
- `count`: show per-file match counts.

FILTERS:
- `path` + `glob`: scope the search root and file pattern.
- A single file path searches that file directly, even if it is hidden or under an ignored root.
- `type`: language/file-type shortcut (py/js/ts/md/json/...).
- default matching uses smart-case: all-lowercase patterns search case-insensitively and patterns containing uppercase stay \
case-sensitive.
- `case_sensitive`: explicitly override smart-case behavior.
- `multiline`: let `.` match newlines and allow multi-line patterns.
- `literal`: search for exact text instead of regex.
- `include_hidden`: include hidden files/directories.
- `include_ignored`: include common dependency/cache roots at workspace root.
- `include_sensitive`: include paths that look like secrets or credentials. Default false.

CONTENT OPTIONS (only for `content` mode):
- `b`: lines before each match.
- `a`: lines after each match.
- `c`: lines before+after and overrides b/a.
- `n`: include line numbers.

LIMITING:
- `offset`: skip the first N result rows/entries.
- `head_limit`: return only first N output rows/entries. Default 250; 0 means unlimited subject to hard caps.

Returns:
- Matching content rows, file paths, or counts according to `output_mode`.
- Truncation metadata such as `content_truncated`, `structured_truncated`, `structured_item_limit`, and \
`structured_char_limit` when output is capped.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Regex pattern to search for, or exact text when `literal=true`. Escape regex "
                            "metacharacters when `literal=false` and searching for literal dots, brackets, "
                            "or file extensions."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional search root or single file path. Use workspace-relative path by "
                            "default; absolute path is allowed when outside-workspace access is enabled. "
                            "Default '.'. A single file path searches that file directly, even if it is "
                            "hidden or under an ignored root."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": (
                            "Optional file glob filter such as `**/*.rs` or `docs/**/*.md`. Use it to narrow "
                            "by filename, path segment, or extension before running broad regex searches. "
                            "Default **/*."
                        ),
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": (
                            "Whether hidden files and dotfiles are included. Default false; set true only "
                            "when explicitly searching hidden project files such as .env.example, .github, "
                            "or dot-directories."
                        ),
                    },
                    "include_ignored": {
                        "type": "boolean",
                        "description": (
                            "When searching workspace root, include files under common "
                            "dependency/cache directories. Default false; set true only when explicitly "
                            "inspecting generated, dependency, cache, or build-output paths."
                        ),
                    },
                    "include_sensitive": {
                        "type": "boolean",
                        "description": (
                            "Include files whose paths look like secrets, credentials, keys, tokens, "
                            "or private config. Default false."
                        ),
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["files_with_matches", "content", "count"],
                        "description": (
                            "Search output mode. `files_with_matches` returns matching paths for a follow-up "
                            "read/search, `content` returns matching lines for inspection, and `count` returns "
                            "per-file match counts. Default is 'files_with_matches'."
                        ),
                    },
                    "literal": {
                        "type": "boolean",
                        "description": "Search for the exact pattern text instead of interpreting it as a regex. Default false.",
                    },
                    "b": {
                        "type": "integer",
                        "description": (
                            "Lines before each match. Only used in content mode. Use when each match needs "
                            "leading context, such as a function signature, heading, import block, or "
                            "preceding error line."
                        ),
                    },
                    "a": {
                        "type": "integer",
                        "description": (
                            "Lines after each match. Only used in content mode. Use when each match needs "
                            "following context, such as a function body, config value, stack trace "
                            "continuation, or adjacent TODO detail."
                        ),
                    },
                    "c": {
                        "type": "integer",
                        "description": (
                            "Context lines before and after each match. Overrides b/a. Use this instead of "
                            "separate b/a values when symmetric context is enough to decide the next read or edit."
                        ),
                    },
                    "n": {
                        "type": "boolean",
                        "description": "Whether to include line numbers in content output. Default true.",
                    },
                    "type": {
                        "type": "string",
                        "description": (
                            "File type shortcut (e.g. py/js/ts/md/json). Unsupported or unknown shortcuts "
                            "return a structured error listing supported values."
                        ),
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of result rows or entries to skip before returning results. Default 0.",
                    },
                    "head_limit": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Cap output to the first N rows or entries. Default 250. Use 0 for unlimited output "
                            "subject to hard caps. Use this for broad searches, then run a narrower follow-up "
                            "query if matches are truncated."
                        ),
                    },
                    "multiline": {
                        "type": "boolean",
                        "description": (
                            "Enable multiline regex mode. Use for patterns that intentionally span line breaks, "
                            "such as adjacent JSON fields, multi-line imports, or repeated prompt blocks."
                        ),
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": (
                            "Explicitly override smart-case behavior. Set true when literal casing matters, "
                            "false when you need forced case-insensitive matching."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    EDIT_FILE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": EDIT_FILE_TOOL_NAME,
            "description": """Safely edit an existing workspace file by replacing exact text.

Use exact `old_string` matching.

Workflow:
- Call `read_file` first unless the file was just fully written with `write_file` or updated by a previous successful \
`edit_file`/`write_file` operation that preserved current context.
- A focused line-range read is enough when your `old_string` comes from that current read state; use a full read for broad or \
uncertain edits.
- Use this for focused edits where a precise old/new string is safer than rewriting a whole file.
- Include enough context in `old_string` to make the target unique; never guess whitespace or punctuation from memory.
- Appending to an unknown existing file does not create a current edit baseline; call `read_file` before editing after that case.
- The operation fails if `old_string` is not found, if it matches multiple locations, or if the file changed since it was read.
- By default `old_string` must match exactly one location; use `replace_all=true` only after confirming every match should change.

Returns:
- Short JSON content with replacement count.
- Structured edit metadata including changed files, bounded diff, additions, deletions, operation, and line ending.
- Clear failure details when the target string is missing, ambiguous, outside the workspace, or rejected by the backend.

Prefer this over shell-based sed/perl for repository edits because it is exact, structured, and safer for Agent-driven \
changes.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Target file path (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact source text to replace. Must be non-empty and unique unless replace_all=true.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": (
                            "Replacement text. May be empty; preserve intended indentation, line endings, "
                            "and surrounding whitespace."
                        ),
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "Replace all matches when true after confirming every match is intended. "
                            "Default false to keep focused edits narrow."
                        ),
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    COMPRESS_MEMORY_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": COMPRESS_MEMORY_TOOL_NAME,
            "description": """Store key summary notes to reduce future context load.

Store a durable memory note that should survive future compaction.

When to use:
- Preserve stable decisions, constraints, file paths, API names, test evidence, user preferences, or implementation facts \
that would be expensive or risky to rediscover.
- Capture facts needed by later turns after a long investigation, a live incident, or a multi-step implementation.
- Use this before context compaction when losing a detail would make the Agent repeat work or make an unsafe assumption.

Good memory notes:
- Include concrete names, paths, commands, identifiers, model names, error text, and final decisions.
- State whether the information is verified current state or only an inference.
- Keep it short enough to be useful but specific enough to resume work.

Do not store transient chatter, obvious facts already present in the latest messages, speculation without labels, secrets, \
or disposable command output.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "core_information": {
                        "type": "string",
                        "description": (
                            "Key information that should be preserved after compression. Include concrete "
                            "names, paths, commands, and decisions when relevant."
                        ),
                    },
                },
                "required": ["core_information"],
            },
        },
    },
    TODO_WRITE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": TODO_WRITE_TOOL_NAME,
            "description": """Create and manage structured TODO list for multi-step execution.

Protocol:
- Send the complete `todos` array each time.
- The payload is a replacement payload, not a patch.
- Existing items with matching `id` are updated.
- Matching items keep their original `created_at`.
- Items omitted from the new array are removed.
- Missing `id` values are generated automatically as short stable ids.
- Each item must include `title`, `status`, and `priority`.
- Missing status defaults to `pending`; missing priority defaults to `medium` at runtime for runtime tolerance.
- Only one item may have `status=in_progress`.

When to use:
- Track multi-step implementation, verification, review, release, or incident recovery work.
- Make progress state explicit before delegating, running long commands, or switching from investigation to edits.

Returns:
- The normalized TODO list with generated ids/defaults and validation errors when statuses conflict.

Use this tool to keep task planning explicit and machine-readable.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": (
                            "Complete TODO list replacement payload. Send every item that should remain; "
                            "omitted existing ids are removed."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": (
                                        "Existing TODO id for update; omit for new item. When omitted, "
                                        "a generated 8-character id is assigned."
                                    ),
                                },
                                "title": {
                                    "type": "string",
                                    "description": (
                                        "TODO title. Make it actionable and observable, so progress can be "
                                        "verified without reading hidden context."
                                    ),
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": (
                                        "TODO status: `pending` for not started, `in_progress` for the single "
                                        "active item, or `completed` after verification."
                                    ),
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                    "description": (
                                        "TODO priority: `high` for blockers or user-critical work, `medium` "
                                        "for normal required work, `low` for cleanup or optional follow-up."
                                    ),
                                },
                            },
                            "required": ["title", "status", "priority"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    BASH_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": BASH_TOOL_NAME,
            "description": """Execute bash command in workspace.

Shell selection:
- By default commands run through a POSIX-style shell (`bash -lc` on Unix-like hosts, `cmd /C` on Windows).
- runtime metadata can override this with `bash_shell` (and Windows shell priority where available).
- runtime metadata `bash_env` can provide extra environment variables for foreground and background commands.
- Returned payloads include the selected shell name so later polling/debugging can match the actual execution environment.

Guidelines:
- Prefer specialized read/write/search/edit tools when possible.
- Use this tool for command execution, package install, scripts, and piped workflows.
- For commands that may prompt for confirmation, pass `auto_confirm=true` or provide explicit `stdin`.
- Use `run_in_background=true` for long-running commands and poll with check tool.
- If a foreground command hits its timeout, it is automatically moved to a background
  session and returns a `session_id` for polling.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command string. The runtime executes it through the configured shell.",
                    },
                    "exec_dir": {
                        "type": "string",
                        "description": (
                            "Execution directory (workspace-relative by default; "
                            "absolute path allowed when outside-workspace access is enabled)."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Foreground timeout seconds, default 300, max 600. For long-running commands, "
                            "prefer `run_in_background=true`; if a foreground command times out it is moved "
                            "to a background session."
                        ),
                    },
                    "stdin": {
                        "type": "string",
                        "description": (
                            "Optional stdin content for interactive prompts, confirmation text, heredoc-style "
                            "input, or commands that read from standard input. Prefer explicit stdin over "
                            "embedding secrets or fragile echo pipelines in the command string."
                        ),
                    },
                    "auto_confirm": {
                        "type": "boolean",
                        "description": (
                            "Pipe yes to the command for non-interactive confirmation prompts. Use carefully: "
                            "do not enable for destructive operations unless the user has already authorized "
                            "the action and the command target is explicit."
                        ),
                    },
                    "run_in_background": {
                        "type": "boolean",
                        "description": (
                            "Run a long-running command in background and return a `session_id`; poll it later "
                            "with `check_background_command` instead of blocking the Agent loop."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    CHECK_BACKGROUND_COMMAND_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": CHECK_BACKGROUND_COMMAND_TOOL_NAME,
            "description": """Check status/output for command launched in background mode, including sessions auto-detached \
after foreground timeout.

When to use:
- After `bash` returns a `session_id` from `run_in_background=true`.
- After a foreground command times out and the runtime auto-detaches it into a background session.
- When a long build, test, server, release, or watcher needs progress checks without blocking the main Agent loop.

Polling protocol:
- Poll until the response is terminal: `completed` or an error such as `background_command_failed`.
- A `running` response can include recent captured stdout/stderr; use that output to decide whether to wait, stop asking for \
status, or report a blocker.
- Stop polling once a terminal status is returned; repeated polling after completion should not be used as a substitute for \
reading the final payload.

Returns:
- Current session status, recent output while running, and final exit/output metadata on completion.
- Structured errors for unknown sessions, failed commands, and runtime-managed background failures.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": (
                            "Background session identifier. It is returned by `bash` when "
                            "`run_in_background=true` or when a foreground command times out."
                        ),
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    CREATE_SUB_TASK_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": CREATE_SUB_TASK_TOOL_NAME,
            "description": """Create sub-tasks for a configured sub-agent.

Modes:
- Single task: provide `task_description` (+ optional `output_requirements`) for one self-contained objective.
- Batch task: provide `tasks` array for multiple independent tasks of the same sub-agent. Use this for parallel work that can \
be split into independent investigations, file reads, reviews, or implementation checks.

Delegation rules:
- Use the exact sub-agent id from the configured `sub_agents` mapping.
- Give the child concrete scope, relevant files or commands, constraints, and expected evidence.
- Do not use batch mode for ordered edits, shared mutable state, dependent tasks, or work where one child result changes what \
the next child should do.
- Keep `include_main_summary=false` for independent tasks; enable it only when the child truly needs parent context.
- `exclude_files_pattern` is a discovery filter over normalized workspace-relative paths. It hides matches from child file \
listing, search, and status snapshots, but direct known-path access, shell commands, and custom tools remain available. It is \
not an access-control boundary or sandbox.

Execution:
- `wait_for_completion=true` (default): wait for result(s) and return final payload. Batch mode may run requests through the \
runtime execution backend in parallel and returns a summary plus one result per task.
- `wait_for_completion=false`: start background sub-task(s) and return `task_id` / `task_ids` for later polling.
- Batch payloads can include partial failures; inspect the summary and each result before deciding whether the parent task \
can continue.

Result handling:
- For synchronous runs, read every returned result and error entry before using the child output.
- Treat partial failures as unresolved work unless the failed child was optional or its failure is itself the required evidence.
- For background runs, preserve the returned task ids and use `sub_task_status` later to inspect progress, fetch results, or \
send follow-up messages.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": (
                            "Exact sub-agent identifier from the configured `sub_agents` mapping. Do not pass "
                            "a display name, model name, or inferred label."
                        ),
                    },
                    "task_description": {
                        "type": "string",
                        "description": (
                            "Single-task description for one self-contained objective. Mutually exclusive with "
                            "`tasks`; give a concrete objective, constraints, relevant files or commands, and "
                            "the evidence or deliverable expected by the parent Agent."
                        ),
                    },
                    "output_requirements": {
                        "type": "string",
                        "description": (
                            "Optional output constraints for single-task mode. State success criteria, expected "
                            "format, concrete deliverables, and verification evidence the parent Agent needs."
                        ),
                    },
                    "tasks": {
                        "type": "array",
                        "description": (
                            "Batch mode: multiple independent tasks for the same sub-agent. Use when parallel "
                            "work can be safely delegated without shared mutable state or ordering dependencies."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": (
                                        "Task description for one independent sub-task. Give a concrete objective, "
                                        "relevant constraints, files or commands, and the evidence or deliverable "
                                        "expected by the parent Agent."
                                    ),
                                },
                                "output_requirements": {
                                    "type": "string",
                                    "description": (
                                        "Optional output constraints for one sub-task. State success criteria, "
                                        "expected format, concrete deliverables, and verification evidence."
                                    ),
                                },
                            },
                            "required": ["task_description"],
                        },
                    },
                    "include_main_summary": {
                        "type": "boolean",
                        "description": (
                            "Whether to include parent-task summary context. Default false. Use when the child "
                            "needs parent context; keep false for independent tasks."
                        ),
                    },
                    "exclude_files_pattern": {
                        "type": "string",
                        "description": (
                            "Optional portable regex applied to normalized workspace-relative paths for child discovery "
                            "only. Matching paths are hidden from file listing, search, and status snapshots. Direct "
                            "known-path access, shell commands, and custom tools remain available, so this is not an "
                            "access-control boundary or sandbox. Blank values are treated as absent; lookaround, "
                            "backreferences, Unicode property escapes, engine-specific anchors/escapes, possessive "
                            "quantifiers, and character-class set operations are rejected."
                        ),
                    },
                    "wait_for_completion": {
                        "type": "boolean",
                        "description": (
                            "Whether to wait for completion. Default true; false starts background execution. "
                            "When false, returned task ids can be polled with `sub_task_status`."
                        ),
                    },
                },
                "required": ["agent_id"],
            },
        },
    },
    SUB_TASK_STATUS_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": SUB_TASK_STATUS_TOOL_NAME,
            "description": """Inspect sub-task status and optionally interact with a sub-task.

Capabilities:
- Query one or more sub-task ids.
- Return lightweight snapshot progress (`detail_level=snapshot`).
- Send `message` to the first task id to steer a running task or continue a completed one.
- Wait for long-running background sub-task completion without repeated polling (`wait_for_completion=true`).
- Optionally wait for the follow-up response with `wait_for_response=true`.

Waiting:
- Use `wait_for_completion=true` when the parent Agent has no useful work until the background sub-task result is available.
- The runtime waits inside this tool call and returns when queried task(s) finish or `max_wait_seconds` is reached.
- Use `check_interval_seconds` as the suggested future re-check interval if the wait reaches its limit.

Continuation rules:
- When `message` is provided, only the first task id is targeted.
- Running tasks receive the message as queued steering input.
- Completed tasks are continued in the same sub-agent session unless they stopped at `max_cycles`.
- Do not continue a child task stopped at `max_cycles`; create a new task with clearer scope or report the child as blocked.
- Use `wait_for_response=true` only when the parent Agent needs the follow-up result before continuing.

Snapshot use:
- Use `detail_level=snapshot` when deciding whether to wait, send a follow-up, or summarize child work.
- Keep `workspace_file_limit` low when file lists add noise; raise it only when files are needed to assess progress.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "description": (
                            "Sub-task ids to query. Use the ids returned by `create_sub_task`; duplicate ids "
                            "are deduplicated. When `message` is provided, only the first id is used as the target."
                        ),
                        "items": {"type": "string"},
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "Optional follow-up or steering message for the first task id. Can steer a running "
                            "task or continue a completed one."
                        ),
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["basic", "snapshot"],
                        "description": (
                            "Status response detail level. `snapshot` includes recent activity, "
                            "latest tool call, and workspace files."
                        ),
                    },
                    "workspace_file_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": (
                            "Maximum number of workspace files returned per task in snapshot mode. Default 20. "
                            "Lower this when file lists add noise; raise it only when files are needed to assess progress."
                        ),
                    },
                    "wait_for_completion": {
                        "type": "boolean",
                        "description": (
                            "Optional. If true and any queried sub-task is still running, wait inside this tool call "
                            "until the task finishes or max_wait_seconds is reached. Use this for long-running "
                            "background sub-tasks when the parent Agent needs the result before continuing."
                        ),
                        "default": False,
                    },
                    "check_interval_seconds": {
                        "type": "integer",
                        "minimum": 30,
                        "maximum": 1800,
                        "description": (
                            "Optional. Used with wait_for_completion=true. Suggested re-check interval in seconds "
                            "if max_wait_seconds is reached while tasks are still running. Default 300."
                        ),
                        "default": 300,
                    },
                    "max_wait_seconds": {
                        "type": ["integer", "null"],
                        "minimum": 60,
                        "maximum": 86400,
                        "description": (
                            "Optional. Used with wait_for_completion=true. The maximum total wait time before "
                            "returning the current still-running status to the Agent. Null or omitted uses "
                            "the system default."
                        ),
                        "default": None,
                    },
                    "wait_for_response": {
                        "type": "boolean",
                        "description": (
                            "When `message` is provided, wait until the task finishes processing that message. "
                            "Use true after sending `message` when the parent Agent needs the follow-up result "
                            "before continuing; keep false for lightweight steering of a still-running child."
                        ),
                    },
                },
                "required": ["task_ids"],
            },
        },
    },
    READ_IMAGE_TOOL_NAME: {
        "type": "function",
        "function": {
            "name": READ_IMAGE_TOOL_NAME,
            "description": """Read image from workspace path or HTTP URL, then attach the image payload to the next LLM turn \
as multimodal content.

When to use:
- Use this before reasoning about image content such as UI screenshots, diagrams, visual errors, generated assets, or visual \
regression evidence.
- Use this when text tools can only tell you that an image exists, but the Agent needs to inspect what is actually visible.
- Prefer workspace-relative paths for local artifacts unless outside-workspace access is explicitly enabled.

Supported inputs:
- Supported formats for workspace files: PNG, JPEG, WEBP, and BMP.
- Inline local image transport is limited to 5 MiB to protect the LLM request size.
- HTTP URLs are passed through as image URLs without downloading.

Returns:
- A multimodal attachment for the next model turn plus normalized source metadata.
- Structured errors for unsupported file types, oversized local images, missing paths, or paths outside the permitted \
workspace.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Image path or URL to attach. Workspace files may be PNG, JPEG, WEBP, or BMP; "
                            "workspace-relative paths are preferred unless outside-workspace access is enabled. "
                            "HTTP URLs are passed through without downloading."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
}

TASK_FINISH_TOOL_SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": TASK_FINISH_TOOL_NAME,
        "description": """When task goals are fully complete, call this tool to end the task and return final message.

Finish the current task and return the final response.

When to use:
- Only call this when the user's requested work is genuinely complete, verified, and no unfinished TODO remains.
- Use it after implementation, review, test output, and any required artifact paths are ready to report.
- Use `exposed_files` to list concrete deliverables the user should inspect.

Completion protocol:
- Do not call this tool if work is partially complete, blocked, waiting for user input, or still needs verification.
- If `todo_write` has pending or in-progress work, the runtime rejects premature finish by default.
- `require_all_todos_completed=false` can bypass the TODO guard only when the user explicitly accepts unfinished TODOs or \
the remaining items are intentionally deferred.
- The message should include concise outcome, important verification evidence, and any remaining caveats.""",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "Final response shown to user. Include the result, important verification evidence, "
                        "and any remaining caveats."
                    ),
                },
                "require_all_todos_completed": {
                    "type": "boolean",
                    "description": (
                        "Default true. When true, finish is rejected while TODO items remain pending or "
                        "in_progress. Set false only when intentionally finishing with remaining TODOs, "
                        "such as when the user explicitly accepts deferred work."
                    ),
                },
                "exposed_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional workspace-relative paths for created or modified deliverables the user should "
                        "inspect. Include concrete artifact paths, not transient logs, prose descriptions, "
                        "or unrelated files."
                    ),
                },
            },
            "required": [],
        },
    },
}

ASK_USER_TOOL_SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": ASK_USER_TOOL_NAME,
        "description": """Pause execution and ask the user for required clarification or decision.

When to use:
- The task cannot be completed safely because a real user preference, permission, credential, destructive action, or \
ambiguous scope decision is missing.
- A reasonable default would risk doing the wrong work or violating the user's stated constraints.
- Multiple clear options exist and user choice changes the implementation or operational outcome.

Do not use this for facts you can discover with available tools, files, command output, documentation, or local \
configuration. This blocks the runtime until the user responds, so keep the question concrete, include 2-3 options when \
possible, and ask only for the decision needed to proceed.""",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Question text to ask the user. Ask the smallest decision needed to unblock progress, "
                        "include relevant context, and avoid bundling unrelated questions."
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional answer options shown to the user. Prefer 2-3 concise, mutually exclusive "
                        "choices when the decision has clear outcomes."
                    ),
                },
                "selection_type": {
                    "type": "string",
                    "enum": ["single", "multi"],
                    "description": (
                        "Single or multi-choice mode when options are provided. Use `multi` only when several "
                        "choices may validly apply at the same time."
                    ),
                },
                "allow_custom_options": {
                    "type": "boolean",
                    "description": (
                        "Whether users can add custom options. Set true when preset options may be incomplete "
                        "or the user may need to provide a custom path, credential label, or preference."
                    ),
                },
            },
            "required": ["question"],
        },
    },
}

ACTIVATE_SKILL_TOOL_SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": ACTIVATE_SKILL_TOOL_NAME,
        "description": """Activate a skill from the current task's available skill list.

When to use:
- A listed skill directly applies to the current task, workflow, domain, or required process discipline.
- The skill may contain repository-specific instructions, validation steps, tool constraints, or templates that should guide \
the next action.

Protocol:
- Use this tool only for skills explicitly listed in <available_skills>.
- Do not invent skill names or activate unrelated skills.
- Read the returned SKILL.md instructions before acting, then follow any mandatory workflow.

The skill metadata follows the Agent Skills specification (https://github.com/agentskills/agentskills): name/description are \
exposed in <available_skills>, and skill instructions are loaded from SKILL.md when location is provided.""",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": (
                        "Skill identifier from available skill list. The exact `name` from the available skill "
                        "list. Do not pass a path, title, or inferred alias."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Optional reason for activating this skill. Explain briefly why this skill applies before acting."
                    ),
                },
            },
            "required": ["skill_name"],
        },
    },
}


def get_default_tool_schemas() -> dict[str, ToolSchema]:
    merged: dict[str, ToolSchema] = {
        TASK_FINISH_TOOL_NAME: deepcopy(TASK_FINISH_TOOL_SCHEMA),
        ASK_USER_TOOL_NAME: deepcopy(ASK_USER_TOOL_SCHEMA),
        ACTIVATE_SKILL_TOOL_NAME: deepcopy(ACTIVATE_SKILL_TOOL_SCHEMA),
    }
    for tool_name in WORKSPACE_TOOLS:
        merged[tool_name] = deepcopy(WORKSPACE_TOOLS_SCHEMAS[tool_name])
    for extra_tool_name in (
        BASH_TOOL_NAME,
        CHECK_BACKGROUND_COMMAND_TOOL_NAME,
        CREATE_SUB_TASK_TOOL_NAME,
        SUB_TASK_STATUS_TOOL_NAME,
        READ_IMAGE_TOOL_NAME,
    ):
        merged[extra_tool_name] = deepcopy(WORKSPACE_TOOLS_SCHEMAS[extra_tool_name])
    return merged
