# vv-agent documentation

These documents are the source of truth for how vv-agent works. They are
versioned with the code, so a reader or a coding agent can rely on the
repository instead of chat history. `README.md` (and `README_ZH.md`) is the
place to start if you are new.

## Using vv-agent

| Document | Read it when you want to |
| --- | --- |
| [architecture.md](architecture.md) | Understand the runtime layout, module boundaries and invariants. |
| [model-settings.md](model-settings.md) | Configure `LLM_SETTINGS`, key files, model defaults and exact model resolution. |
| [token-counting.md](token-counting.md) | Understand local compaction estimates, incremental counting and their limits. |
| [runtime-control.md](runtime-control.md) | Wait for and resume replies, approvals, background work, cancellation and events. |
| [run-budgets.md](run-budgets.md) | Limit tokens, tool calls, wall time or host cost for a run. |
| [output-validation.md](output-validation.md) | Validate a typed final output and allow one tools-free repair. |
| [bash-process-management.md](bash-process-management.md) | Run, query and stop shell commands, including background ones. |

## Hosting vv-agent

| Document | Read it when you want to |
| --- | --- |
| [host-integration.md](host-integration.md) | Run sessions on many workers against one PostgreSQL database. |
| [session-model-selection.md](session-model-selection.md) | Let users switch the model of a running session. |
| [upgrading.md](upgrading.md) | Move a host with stored sessions to a newer release. |
| [app-server.md](app-server.md) | Talk to vv-agent over the JSONL App Server protocol. |
| [app-server-host-integration.md](app-server-host-integration.md) | Embed the App Server in a desktop app, IDE or service. |
| [app-server-runtime-mapping.md](app-server-runtime-mapping.md) | See how App Server threads, turns and items map onto sessions. |

## Working on vv-agent

| Document | Read it when you want to |
| --- | --- |
| [development.md](development.md) | Set up, test, lint and run live tests. |
| [session-kernel.md](session-kernel.md) | Change the session kernel: log, inbox, leases, recovery, stores. |
| [parity-contract.md](parity-contract.md) | Change behavior governed by `vv-agent-contract`. |
| [session-kernel-capability-matrix.md](session-kernel-capability-matrix.md) | Find the module and test behind each runtime capability. |
| [session-kernel-baseline.md](session-kernel-baseline.md) | Check or reproduce the kernel performance measurements. |

## Releases

Release notes live in [releases/](releases/): [0.30.4](releases/0.30.4.md), [0.30.3](releases/0.30.3.md), [0.30.2](releases/0.30.2.md), [0.30.1](releases/0.30.1.md), [0.30.0](releases/0.30.0.md), [0.29.0](releases/0.29.0.md), [0.28.0](releases/0.28.0.md), [0.27.1](releases/0.27.1.md), [0.27.0](releases/0.27.0.md), [0.26.0](releases/0.26.0.md), [0.25.0](releases/0.25.0.md), [0.24.0](releases/0.24.0.md),
[0.23.1](releases/0.23.1.md), [0.23.0](releases/0.23.0.md),
[0.22.0](releases/0.22.0.md). Store and API changes that hosts must act on are
collected in [upgrading.md](upgrading.md#version-notes).

## Other entry points

- `examples/README.md`: runnable examples.
- `local_settings.example.py`: settings template with placeholder keys.
- `pyproject.toml`: package metadata, dependency groups, pytest markers and
  lint settings.
- `tests/`: the executable behavior contract.

## Keeping docs useful

- Put a change in the one document that owns the behavior, and link to it from
  elsewhere instead of restating it.
- Describe how things work now. Test counts, gate results and adoption status
  belong in release notes and the contract support matrix, not in reference
  docs.
- Keep `AGENTS.md` short; it points here.
- Use commands that run from the repository root and relative paths such as
  `src/vv_agent/config.py`.
- If a documented behavior can drift, point to the test that pins it.
