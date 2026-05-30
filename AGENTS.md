# AGENTS.md

This file is a map, not the full manual. Keep it short and use `docs/` as the
versioned source of truth for architecture, workflows, and model configuration.

## Start Here

- Read `docs/index.md` first when the task touches more than one file.
- Use `docs/architecture.md` for runtime, SDK, tools, memory, backend, and
  workspace boundaries.
- Use `docs/development.md` for setup, commands, test selection, linting, and
  release hygiene.
- Use `docs/model-settings.md` before changing model defaults, settings
  parsing, live tests, or examples that mention provider model ids.

## Repository Rules

- Work from the repository root.
- Use `uv run ...` for Python commands so the repo-managed environment is used.
- Do not edit or commit `local_settings.py`; it may contain real keys. Use
  `local_settings.example.py` for checked-in templates.
- Do not add model aliases for independent provider models. If a model key is
  requested, resolve that exact key from `LLM_SETTINGS`.
- Keep user-facing defaults aligned across CLI, README, examples, tests, and
  `local_settings.example.py`.
- After significant behavior changes, update the relevant docs under `docs/`
  and then keep this file as a pointer only.

## Common Commands

```bash
uv sync --dev
uv run ruff check
uv run pytest
```

Targeted checks are preferred while iterating:

```bash
uv run pytest tests/test_config.py
uv run pytest tests/test_sdk_client.py
uv run pytest tests/test_runtime.py
```

Live tests are opt-in and require real provider credentials:

```bash
V_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live
```

## Change Boundaries

- Config parsing lives in `src/vv_agent/config.py`.
- CLI defaults live in `src/vv_agent/cli.py`.
- Runtime orchestration lives under `src/vv_agent/runtime/`.
- Tool schemas and handlers live under `src/vv_agent/tools/`.
- SDK entry points live under `src/vv_agent/sdk/`.
- Workspace storage backends live under `src/vv_agent/workspace/`.

When a change crosses these boundaries, add or update tests in the matching
`tests/test_*.py` module before reporting completion.
