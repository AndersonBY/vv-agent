# Bash process management

The built-in `bash` tool runs a shell command in the workspace. If the command
is still running after a short wait, the tool returns a handle and the command
keeps running in the background. Two more tools work with that handle:
`check_background_command` and `stop_background_command`.

## The `bash` tool

`bash` takes these arguments:

| Argument | Meaning | Range and default |
| --- | --- | --- |
| `command` | Command to run through the configured shell | Required |
| `exec_dir` | Directory to run in | Defaults to the workspace. Workspace-relative; absolute paths only when outside-workspace access is enabled |
| `stdin` | Text written to the command's standard input | Optional |
| `auto_confirm` | Pipe `yes` into the command for confirmation prompts | Default `false` |
| `yield_time_ms` | How long to wait for the command before returning. Reaching it never kills the command | Integer 0..10000, default 1000 |
| `timeout_seconds` | Execution deadline, counted from process start and including time after a handle is returned | Optional integer 1..86400; omitted means no deadline |

For example, this starts a local HTTP server that is killed after one minute:

```json
{"command":"python -m http.server 8000 --bind 127.0.0.1","yield_time_ms":1000,"timeout_seconds":60}
```

If the command exits within `yield_time_ms`, the tool returns its output. If
it is still running, the result includes `status=running`, a `session_id`
handle, and the output so far. That result has status `SUCCESS` with the
`continue` directive: starting the command is done, and the runner moves on to
the next model cycle. The tool result is recorded at that point, and the
command's later outcome never changes it. Use `yield_time_ms=0` to get a
handle immediately.

Writing `stdin` does not wait for the command to read it.

## Checking and stopping a command

Both management tools take only the handle:

```json
{"session_id":"bg_0123456789ab"}
```

- `check_background_command` returns the command's current state and output
  right away.
- `stop_background_command` asks for the whole process tree to be terminated.

Checking a command or returning its handle never resets the deadline. The
process manager enforces `timeout_seconds` even if nobody checks the command.

Each handle belongs to the task that started it and to that task's local
workspace root. Both tools check ownership before they poll, read output,
write artifacts, or send signals. Knowing an artifact path does not grant
access to the command.

The process manager keeps its records in memory. After a restart, a handle
reports `missing`. That means the runtime can no longer observe the command,
not that the process has exited.

## Results and exit codes

Final results keep the real exit code.

- A nonzero exit, including death by a signal, is an `ERROR`.
- A timeout is an `ERROR`, even if the command handles the termination signal
  and exits with code 0.
- A stop is final only when the runtime has confirmed that no process in the
  tree is still running. Until then, the result reports `stopping` or
  `unknown`: a successful observation with no exit code. Check again later.

Interactive watchers are notified when a command reaches a confirmed final
state.

## Platform support and supervision

Full process-tree tracking works on Linux only.

On Linux, each command gets its own small supervisor process, started with
the current Python interpreter and using only the standard library. The
supervisor marks itself as a subreaper, so orphaned descendants are reparented
to it. The application itself never becomes a subreaper and never reaps
unrelated children. The supervisor keeps track of descendants that call
`setsid()` or double-fork, even after the original parent exits. It reports
the command finished only after every descendant has been reaped, and then
returns the original command's own exit code or signal. If the supervisor is
lost before it sends that report, the handle stays `unknown` and its captured
output is kept.

The supervisor must report ready within five seconds. If it does not, or
sends a partial message, or its startup report is lost, the runtime asks it to
clean up but keeps an `unknown` handle with its captured output. The runtime
does not claim that the command never started or that its tree has stopped. If
the ready message arrives late, the runtime reads it before the final report,
so a later check can still return the final result. If the owner disconnects
during startup, the supervisor reaps its own command tree. This startup limit
is separate from the command's `timeout_seconds`.

On macOS and Windows, commands still launch, accept stdin, return output
snapshots, and get the platform's usual termination attempts. There is no
reliable way to confirm that a whole tree has exited, though. Even a short
command whose parent exited can stay `unknown`, and a stop can stay
`stopping`. Neither counts as success and neither has an exit code. Use Linux
when a workflow needs confirmed completion or confirmed stop. The same limit
applies to an externally created `Popen` process handed to the manager: the
parent's exit alone does not prove that its tree is gone.

## Output and artifacts

While a command runs, its output is shown as a Unicode head and tail preview.
Output that is too large goes to the artifact store:

- Each check that produces an artifact freezes the output captured up to that
  moment. `read_file` on the artifact returns the complete text.
- Later output never changes an earlier artifact.
- Repeated checks of a finished command reuse the same final artifact.
- If saving the artifact fails, the result does not claim the output is
  recoverable.

If reading captured output fails, the owner can retry. A later successful
read clears the error. A retry never sends a second completion notification
and never deletes a new file that now uses an old capture path.

Copying live output and writing artifacts happen outside the session state
lock, so a slow storage backend cannot stop the watchdog from enforcing the
deadline. If the capture becomes unavailable while the command finishes, the
result is an output error. That error does not reset the deadline or allow the
command to run again.

## Tests

`tests/test_bash_process_management.py` runs real child processes, including
a loopback HTTP server, through the tool registry and a SQLite session-backed
runner with `ScriptedLLM`. `tests/test_bash_tools.py` covers the basic shell,
stdin, and artifact cases.

```bash
uv run --no-sync pytest tests/test_bash_process_management.py tests/test_bash_tools.py tests/test_builtin_tool_behavior_contract.py
```
