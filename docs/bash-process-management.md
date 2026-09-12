# Bash Process Management

The built-in tools use Contract 21. `bash` accepts `command` (required),
`exec_dir` (workspace by default), optional text `stdin`, `auto_confirm`
(false by default), `yield_time_ms` and `timeout_seconds`.

| Parameter | Meaning | Range and default |
| --- | --- | --- |
| `yield_time_ms` | Initial observation wait; reaching it never kills the command | Integer 0..10000, default 1000 |
| `timeout_seconds` | Execution deadline from process start, including time after returning a handle | Optional integer 1..86400; omission means no deadline |

For example, start a local HTTP service with a one-minute execution limit:

```json
{"command":"python -m http.server 8000 --bind 127.0.0.1","yield_time_ms":1000,"timeout_seconds":60}
```

If it remains running, the receipt includes `status=running`, `session_id` and
bounded output. The receipt is `SUCCESS` with `continue`: the management action
is complete, and a checkpointed Runner proceeds to its next model cycle.
The later process outcome never rewrites that recorded result or its digest.
Use `yield_time_ms=0` to request an immediate handle. Both management tools
accept only that handle:

```json
{"session_id":"bg_0123456789ab"}
```

Call `check_background_command` to read an immediate current snapshot, or
`stop_background_command` to request process-tree termination. Querying and
returning a handle never reset the execution clock. The existing local manager
enforces an explicit deadline even without subsequent queries. Stdin delivery
does not wait for the child to consume the supplied input.

The manager binds each handle to the initiating task ID and canonical local
workspace root. Query and stop validate ownership before polling, reading
output, writing artifacts or signaling. Artifact path labels do not grant
access. Restarting the process-local manager loses its records: `missing`
means the local observation is unavailable, not that an external process died.

Terminal results retain actual exit codes. A nonzero exit is `ERROR`, including
a process stopped by a signal; timeout is an error even if the child handles
termination by exiting zero. A stop is terminal only after the managed process
tree is confirmed to have no executing members. `stopping` or `unknown` returns
a successful observation with no exit code while termination is unconfirmed.
Existing interactive watchers receive confirmed terminal notifications.

## Platform support and supervision

The complete local process-tree lifecycle in this release is supported on Linux.
Each command has a separate, stdlib-only supervisor started with the current
Python interpreter. Only that process becomes a subreaper; the application does
not adopt or reap unrelated children. The supervisor retains descendants that
call `setsid()` or double-fork, including after the original parent exits. It
confirms completion only when every descendant has been reaped, then preserves
the original command's actual exit or signal. Losing the supervisor without its
completion proof leaves an `unknown` session and retains the capture.

Supervisor readiness has a five-second absolute handshake limit. A timeout,
partial packet, or lost startup observation requests cleanup but retains an
owner-bound `unknown` handle and its capture; it does not claim the command was
never started or that its tree stopped. A late ready packet is consumed before
the terminal proof, so a later query can recover the final result. Disconnecting
the owner during startup also makes the supervisor reap its own command tree.
This is a supervisor-startup failure policy, separate from the execution deadline.

macOS and Windows retain command launch, stdin, output snapshots, and their
existing platform termination attempts, but this implementation has no reliable
complete-tree proof there. Even a short command whose parent exited can remain
`unknown`, and a stop can remain `stopping`; neither is a terminal success and
neither supplies an exit code. This is an explicit availability limitation from
requiring tree confirmation. Workflows that need confirmed completion or stop
must use Linux for this release. Do not describe the local C21 lifecycle as
fully supported on all platforms. Adopting an externally created, unsupervised
`Popen` also cannot prove its complete tree from the parent's exit alone.

Running output is a current Unicode head/tail preview. Oversized output uses
the existing artifact store: each live artifact freezes the exact captured
prefix from that query, and `read_file` can recover the complete text. Later
output does not modify an earlier artifact. Repeated terminal queries reuse
one retained terminal artifact. Persistence failures never claim recoverability.
Failed live or terminal capture reads are retryable by the same owner. A later
successful read clears the error; retrying does not emit a second completion
notification or delete a new file at an already released capture path.

`tests/test_bash_process_management.py` exercises the real registry, SQLite
checkpointed Runner, ScriptedLLM, real children and a loopback HTTP service.
It covers immediate yield, elapsed yield, running queries, a mixed tool batch,
receipt immutability, ownership denial, deadlines, stop, Unicode and artifacts.
It also covers detached and double-fork children, original signal preservation,
startup timeouts and partial packets, disconnected readiness, lost supervision,
unrelated-child/socket isolation, and output-read recovery.
`tests/test_bash_tools.py` retains the original shell, stdin and artifact cases.

```bash
uv run --no-sync pytest tests/test_bash_process_management.py tests/test_bash_tools.py tests/test_builtin_tool_behavior_contract.py
```

## Live output and deadlines

Live-output copying and artifact writes use a captured observation outside the
session state lock. A slow storage backend does not prevent the watchdog from
enforcing the command's original execution deadline. An observation whose
capture became unavailable during completion is reported as an output error;
it does not reset the deadline or authorize a duplicate command.
