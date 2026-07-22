# AGENTS.md

This file is a map, not the full manual. Keep it short and use `docs/` as the
versioned source of truth for architecture, workflows, and model configuration.

## Start Here

- Read `docs/index.md` first when the task touches more than one file.
- Use `docs/architecture.md` for runtime, SDK, tools, memory, backend, and
  workspace boundaries.
- Read `contract.lock.json` and `docs/parity-contract.md` before changing any
  public, model-visible, runtime, persistence, or wire behavior shared with
  `vv-agent-rs`.
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

## Cross-Language Parity

- Canonical shared behavior lives in sibling `../vv-agent-contract/` and its
  versioned GitHub releases. This repository and `../vv-agent-rs/` are two
  implementations, not independent contract sources.
- `contract.lock.json` pins the exact version, Git revision, release artifact,
  and fixture digest. Read it before parity work.
- `tests/fixtures/parity/` is a generated vendored snapshot. Never edit it
  directly; update `vv-agent-contract/` first and run
  `scripts/contract_snapshot.py sync`.
- Follow `vv-agent-contract/docs/change-workflow.md` for classification,
  paired adoption, status transitions, and cross-repository gates.
- Model-visible prompts and built-in tools, public defaults, errors, side
  effects, cancellation, persistence, events, App Server protocol, and wire
  fixtures require paired implementation and behavior tests.
- Language-idiomatic API spelling is allowed only when both sides can express
  the same input, observe the same output, and enforce the same safety boundary;
  record the shared rule centrally and the Python mapping in
  `docs/parity-contract.md`.
- Do not mark a version `verified` until both locks select the same contract,
  both real producer suites and full gates pass, and central cross-repository CI
  records both implementation revisions.
- If the sibling repository cannot be updated in the same change, record an
  explicit open parity gap and do not report the shared feature complete.
- Keep `HEAD` forward-only. Maintain one current public and wire shape, and
  delete superseded readers, aliases, shims, migrations, fixtures, tests, and
  documentation in the same paired change. Git tags provide old runtimes.
- Backward compatibility is not a design or acceptance requirement. Prefer a
  breaking replacement when it improves the current architecture, update active
  callers in the same change, and leave old behavior only in pinned releases.
- Schema and protocol versions are strict rejection boundaries, not decoder
  selectors. Reject missing, stale, unknown, and malformed versions, and reject
  unknown fields unless the central contract defines a typed extension map.

## Common Commands

```bash
uv sync --dev
python3 scripts/contract_snapshot.py check
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
VV_AGENT_RUN_LIVE_TESTS=1 uv run pytest -m live
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
