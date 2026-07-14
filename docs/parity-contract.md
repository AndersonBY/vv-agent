# Python Contract Integration

`vv-agent` implements the Python side of the canonical contract published by
[`AndersonBY/vv-agent-contract`](https://github.com/AndersonBY/vv-agent-contract).
The normative behavior and change workflow no longer live in this repository.

## Pinned Contract

`contract.lock.json` is the machine-readable adoption record. It pins:

- semantic contract version;
- exact central Git revision;
- immutable release artifact URL and SHA-256;
- local vendored snapshot path;
- canonical `SHA256SUMS` digest.

`tests/fixtures/parity/` is generated from that release. It is committed for
offline and reproducible tests, but it is not an editable source of truth.

## Required Reading

For shared public, model-visible, runtime, persistence, or wire changes, read:

1. `contract.lock.json` in this repository;
2. `../vv-agent-contract/AGENTS.md`;
3. `../vv-agent-contract/docs/parity-contract.md`;
4. `../vv-agent-contract/docs/change-workflow.md`;
5. sibling `../vv-agent-rs/docs/parity-contract.md`.

If the sibling checkout is unavailable, use the exact repository and revision
from the lock. Do not infer the current contract from a floating `main` branch.

## Snapshot Commands

Offline verification of the committed snapshot:

```bash
python3 scripts/contract_snapshot.py check
```

Stronger verification against the sibling canonical checkout:

```bash
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
```

Synchronization is allowed only after the canonical version is committed and
its deterministic release zip exists:

```bash
python3 scripts/contract_snapshot.py sync \
  --source ../vv-agent-contract \
  --artifact /path/to/vv-agent-contract-<version>.zip \
  --artifact-url https://github.com/AndersonBY/vv-agent-contract/releases/download/v<version>/vv-agent-contract-<version>.zip
```

Never repair a contract failure by editing a file under
`tests/fixtures/parity/` or changing only a digest.

## Python Producer Map

| Contract surface | Python producer or evidence |
| --- | --- |
| Public API inventory | `src/vv_agent/__init__.py`, `tests/test_parity_evidence_manifests.py` |
| System prompt | `src/vv_agent/prompt/`, `tests/test_prompt_builder.py` |
| Built-in tool specification | `src/vv_agent/constants/workspace.py`, `src/vv_agent/tools/registry.py`, `tests/test_tool_schema_contract.py` |
| Agent, Runner, result, live control | `src/vv_agent/agent.py`, `src/vv_agent/runner.py`, `src/vv_agent/run_handle.py` |
| Delegation and background tasks | `src/vv_agent/background_task.py`, `src/vv_agent/handoffs.py`, `src/vv_agent/runtime/sub_task_manager.py` |
| Sessions and stores | `src/vv_agent/sessions/`, `src/vv_agent/runtime/stores/`, `tests/test_session_store_parity.py` |
| Events and tracing | `src/vv_agent/events.py`, `src/vv_agent/event_store.py`, `src/vv_agent/tracing.py` |
| Token and cache usage | `src/vv_agent/types.py`, `src/vv_agent/runtime/token_usage.py`, `src/vv_agent/llm/vv_llm_client.py`, `tests/test_token_usage_contract.py` |
| Distributed runtime | `src/vv_agent/runtime/backends/distributed.py`, `src/vv_agent/runtime/checkpoint_codec.py` |
| App Server | `src/vv_agent/app_server/`, `tests/test_app_server_contract_parity.py` |
| Real closure tests | `tests/test_parity_evidence_manifests.py`, `tests/test_runner_events_producer_parity.py` |

A fixture parser or private helper test cannot replace a real public producer
test. A field that is declared but ignored by a planner, executor, provider, or
store remains a contract failure.

## Python Adaptations

The following are API-shape adaptations, not behavioral differences:

- dataclasses, protocols, decorators, and exceptions map to Rust structs,
  traits, builders, and `Result`;
- synchronous convenience wrappers may coexist with async internals;
- Python `output_type` coercion maps to Rust typed deserialization;
- Celery adapters map to Rust Apalis adapters through the same distributed
  envelope, checkpoint, lease, and terminal-state contract;
- Python settings-file model controls map to the equivalent Rust
  `ModelProvider` capability.

Add a new adaptation only when both implementations preserve input, output,
safety, persistence, cancellation, and lifecycle semantics.

## Completion Gate

```bash
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
uv run pytest
uv run ruff check .
uv run ty check
```

Then run the Rust gate and the central
`vv-agent-contract/.github/workflows/cross-repository.yml` workflow with exact
contract, Python, and Rust refs. If either implementation is incomplete, keep
the central support matrix at `pending-adoption` or `in-progress`.
