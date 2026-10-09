# vv-agent Documentation Index

`vv-agent` keeps project knowledge in versioned Markdown so maintainers and
coding agents can read the repository directly instead of relying on chat
history.

## Core Documents

| Document | Use it for |
| --- | --- |
| `architecture.md` | Runtime structure, module boundaries, execution flow, and invariants. |
| `parity-contract.md` | Python producer mapping and local adoption commands for the canonical `vv-agent-contract` release. |
| `development.md` | Local setup, test commands, linting, live-test workflow, and change hygiene. |
| `model-settings.md` | `LLM_SETTINGS`, local key files, model defaults, and exact model resolution rules. |
| `session-kernel-capability-matrix.md` | F3 capability gate, Runner/SQLite parity evidence, explicit gaps and short-run benchmark methodology. |
| `session-kernel-f2d-children-report.md` | F2d-3 child SDK adapters, host bindings, intentional differences and measured gates. |
| `session-kernel-f2d-surfaces-report.md` | F2d-4 private interactive/CLI/App Server assembly, C1 wire items, F3 deletion inventory and measured gates. |
| `session-kernel.md` | Default session kernel, SQL stores, recovery, compaction, and test isolation. |
| `runtime-control.md` | Background tasks, interrupted-result resume, approvals, sessions, cancellation, and typed event producers. |
| `bash-process-management.md` | Bash initial wait, execution deadline, owner-scoped query/stop, live output, and retained operation receipts. |
| `run-budgets.md` | Token, tool, wall-time, and host-cost limits; retained observations and same-turn resume. |
| `output-validation.md` | Default-off typed output validation, one tools-free repair callback, and failure semantics. |
| `session-kernel-f3a-report.md` | Contract v24 adoption, entrypoint cutover, validation and F3b extraction inventory. |
| `app-server.md` | JSONL protocol, lifecycle, approval, schema generation, CLI startup, and host boundary. |

## Existing Entry Points

- `README.md` and `README_ZH.md`: user-facing usage guide.
- `examples/README.md`: runnable example catalog.
- `local_settings.example.py`: checked-in settings template with placeholder
  keys.
- `pyproject.toml`: package metadata, dependency groups, pytest markers, and
  lint configuration.
- `tests/`: mechanical behavior contract for runtime, SDK, config, tools,
  workspace backends, memory, and live smoke tests.

## Documentation Maintenance

- Update the narrowest document that owns the changed behavior.
- Keep `AGENTS.md` concise; add deeper details here instead.
- Prefer command snippets that can be run from the repository root.
- Avoid hard-coded machine paths. Use relative paths such as
  `src/vv_agent/config.py` or `tests/test_config.py`.
- If a doc describes a behavior that can drift, add or point to a test that
  enforces it.
