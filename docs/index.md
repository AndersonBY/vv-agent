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
| `runtime-control.md` | Background tasks, interrupted-result resume, approvals, sessions, cancellation, and v1 event producers. |
| `run-budgets.md` | Token, tool, wall-time, and host-cost limits; observations, resume, and distributed behavior. |
| `output-validation.md` | Default-off typed output validation, one tools-free repair callback, and failure semantics. |
| `checkpoint-resume.md` | Opt-in checkpoint v2, immutable run definitions, operation journals, ambiguity, and durable terminal ordering. |
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
