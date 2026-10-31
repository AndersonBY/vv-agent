# Python contract integration

`vv-agent` is the Python implementation of a language-neutral contract that
lives in [`AndersonBY/vv-agent-contract`](https://github.com/AndersonBY/vv-agent-contract).
That repository owns the normative behavior, the fixtures, the versioning rules
and the adoption workflow. This page explains how this repository consumes it.

## Pinned contract

`contract.lock.json` pins one contract release. It records:

- the contract version and Git revision;
- the immutable release artifact URL and its SHA-256 digest;
- the path of the vendored fixtures (`tests/fixtures/parity/`);
- the digest of the canonical fixture manifest.

`tests/fixtures/parity/` is a generated copy of the contract fixtures. Do not
edit a vendored fixture or digest by hand; resync it instead (see below).

This page does not track adoption status. The machine-readable source for that
is [`support-matrix.json`](https://github.com/AndersonBY/vv-agent-contract/blob/main/support-matrix.json)
in the contract repository. It lists the required implementations, the
implementation revision each one was verified at, the verification time, the
central CI run URL, and any frozen baselines. The support matrix uses schema 2,
which names `required_implementations` explicitly and records a
`contract_version` per implementation. The tooling rejects schema 1.

Python is the only required implementation. The Rust implementation
(`vv-agent-rs`) is frozen at contract 23.0.0 and package series 0.21.x. It gets
maintenance fixes only and does not follow new contract releases. A newer
contract version, or a new Python verification run, says nothing about Rust.
Bringing Rust back needs a new maintainer decision and full adoption of the
contract that is current at that time, through the central workflow.

## Required workflow

Follow these steps for any change that affects the public API, what the model
sees, runtime behavior, persisted data, or a wire format:

1. Read `contract.lock.json` in this repository and `../vv-agent-contract/AGENTS.md`.
2. Read the parity, versioning and change-workflow documents in the contract repository.
3. Change the canonical docs and fixtures in `vv-agent-contract` first.
4. Sync the vendored snapshot with `scripts/contract_snapshot.py sync`. Leave the
   frozen Rust lock and fixtures untouched.
5. Update the real Python producers and their behavior tests. Updating only the
   fixture parsers is not enough.
6. Run the full Python test and lint gates, then the central cross-repository CI
   with the contract and Python revisions. Rust adoption and Rust gates are not
   required.

## Local unreleased work

For unreleased contract work, build the canonical worktree into a deterministic
ZIP under `/tmp`. Run `scripts/contract_snapshot.py sync` with that local file
and the absolute `/tmp/...` path as `--artifact-url`. The revision is the worktree's base commit;
the artifact and manifest digests pin the uncommitted fixture bytes. This is a
local development lock, not an immutable release pin. Replace it with the
reviewed release revision and URL before publishing a package. Keep adoption
pending until the required central gates pass.

## Snapshot commands

Check the lock and the vendored snapshot:

```bash
python3 scripts/contract_snapshot.py check
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
```

The second form also compares the snapshot byte-for-byte with a local checkout
of the contract repository.

Resync the snapshot once an immutable contract release exists:

```bash
python3 scripts/contract_snapshot.py sync \
  --source ../vv-agent-contract \
  --artifact https://github.com/AndersonBY/vv-agent-contract/releases/download/v<version>/vv-agent-contract-<version>.zip \
  --artifact-url https://github.com/AndersonBY/vv-agent-contract/releases/download/v<version>/vv-agent-contract-<version>.zip \
  --revision <contract-revision>
```

Before a package release, the release workflow requires that the support
matrix lists Python as verified at a revision contained in the release commit:

```bash
python3 scripts/contract_snapshot.py adoption --implementation python --revision "$(git rev-parse HEAD)"
```

## What the tests prove

Fixture and snapshot checks prove only that the vendored files are intact. The
behavior comes from producer tests: tests that run the real Python code and
compare its output with the contract. They cover the public API, runtime
decisions, events, strict wire readers and recovery under the pinned contract.
Producer and store tests run the execution rules against both SQLite and
PostgreSQL.

`scripts/session_kernel_fixtures.py` regenerates the session kernel fixtures
from the real producers. `tests/session/test_session_kernel_fixtures.py`
compares that output byte-for-byte with the vendored snapshot. The snapshot
integrity check in `contract_snapshot.py` runs separately.

An implementation counts as verified only after its producer tests and the
central gates pass. Passing fixture checks alone is not verification.

## Python producer map

Paths are relative to `src/vv_agent/` and `tests/`.

| Surface | Producer | Tests |
| --- | --- | --- |
| Public API v15 | package exports, `Runner`, `RunConfig`, interactive sessions, App Server | `test_parity_evidence_manifests.py` |
| Strict wire formats | `events.py`, `types.py`, `message_codec.py`, `app_server/protocol/` | event validation, protocol type and session codec tests |
| Prompt and run definition | `runtime/compiler.py`, `prompt/`, `session/runtime.py` | `test_run_definition_producer.py`, prompt and session fixture tests |
| Execution and recovery | `session/kernel.py`, `session/sql.py`, `session/reducer.py` | `tests/session/`, App Server durable resume tests |
| Events, results and tracing | `session/projection.py`, `session/result.py`, `session/tracing.py` | Runner event, terminal, trace and result tests |
| Child sessions and handoff | `session/delegation.py`, `session/children.py` | configured child, continuation, agent tool and handoff tests |
| Memory and compaction | `session/compaction.py`, `session/memory.py`, `memory/` | memory lifecycle, local compaction and token usage tests |
| Budgets | `budget.py`, `session/kernel.py` | `test_run_budget.py`, budget and session fixtures |
| Model selection | `session/kernel.py`, `session/runtime.py`, `session/app_server.py` | `tests/session/test_model_selection.py`, `test_app_server_model_selection.py` |
| Provider usage | `session/kernel.py`, `session/providers.py` | `tests/session/test_provider_usage.py`, `tests/session/test_recovery_matrix.py` |
| Session supervision and dispatch | `session/supervisor.py`, `session/children.py`, `session/kernel.py` | `tests/session/test_batch_children.py`, `test_supervisor.py`, `test_dispatch.py` |
| App Server v5 | `app_server/`, `session/app_server.py` | schema, protocol, turn, approval, replay and lifecycle tests |
| App Server result reads | `session/tool_results.py`, `session/tool_result_read.py`, `workspace/resources.py` | `session/test_app_server_tool_results.py`, `app_server_tool_results.json` |
| Workspace and tool results | `tools/`, `workspace/`, `runtime/tool_results.py` | workspace parity, Bash, custom tool and metadata tests |

The workspace edit producer is `tools/handlers/workspace_io.py`. It builds the
canonical edit receipt and file metadata. `test_builtin_tool_behavior_contract.py`
checks it against the central success fixture. The workspace tool tests cover
consecutive edits on large files, stale baselines, exact replacement, and
preservation of a UTF-8 BOM and CRLF line endings.

## Current boundaries

- RunEvent v6, model-call v2, task-token-usage v3 and the strict `Message` shape
  each accept exactly one current version. A missing, older, unknown or
  malformed version is rejected. There is no decoder for contract 23 data. See
  [upgrading.md](upgrading.md#session-data-and-upgrades) for what this means for
  stored data.
- Results carry `session_id` and `turn_id`.
- The App Server speaks protocol v5 and reports thread state with one
  `ThreadStatus` enum.
- The session log is the source of truth for execution. Events, results, the
  App Server and tracing are projections built from it. JSONL event stores only
  receive copies.
- Recovery within the same turn reuses the prompt, logical model, limits and usage that
  were frozen when the turn started.
- Durable stores are opt-in through the session API.
- Inbox schema 3 requires provider usage that the provider has authenticated.

## Model selection

Contract 27 adds session model selection. A host pushes
`InboxItem(kind="model_selection", payload={"selection": ...})` with
`SessionTx.push`. The kernel passes the retained request to
`Runtime.model_resolver`, which returns `(LLMClient, ResolvedModelConfig)`.
`Runtime.resolve_model` then checks the logical model identity against the binding
recorded in the log. App Server clients use `AppServerClient.select_model` with
`ThreadModelSelectParams`, which sends `thread/model/select`.

[session-model-selection.md](session-model-selection.md) describes the full
rules.

## App Server A1

Contract 28 replaces the wire with v3: stable submission receipts, controls and
approvals across connections, bounded record cursor replay and rich status.
`SessionDriver.model_resolver` binds AppServerHost.resolve_model; its provider
default is also used by CLI and interactive sessions. Read ranges use indexed
projection dependencies. The SQLite and PostgreSQL DDL, record schema 1 and
inbox schema 3 stay unchanged. This release is not a store boundary.
Embedded hosts use AppServer.wake(thread_id) after commit. The built-in host
retries blocked leases at expiry, deferring to renewed or replacement holders
until their new expiry. Child delivery wakes the parent after commit. Node observers use independent
per-session backoff without holding store locks across projection or transport I/O.
The public API inventory records the wake entry point; Rust remains frozen.

## Endpoint routing

Contract 29 keeps endpoint IDs out of model and definition identity. Python
uses `Runtime.resolve_model` to obtain current primary clients and
`VvLlmClient.ordered_targets` to freeze each operation's routing. Historical
endpoint fields remain readable evidence. Record schema 1 and inbox schema 3
are unchanged. See [model selection](session-model-selection.md#resolver-contract).

## A2 context mapping

Contract 30.0.0 uses App Server v4 and public API inventory v14. Python maps
thread/context/read to session.context_view and shares session.compaction's
memory_settings resolver with execution. thread/compact/start writes the existing
content extension and feeds summary_requests; it adds no inbox kind or store
version. Typed client methods are read_context and compact_thread. The canonical
A2 vectors are reproduced by scripts/app_server_a2_fixtures.py.

### Completed-boundary forks

Contract 30.1.0 adds thread/fork under App Server v4 without a store boundary.
ThreadForkParams maps to AppServerClient.fork_thread. The canonical rules are in
the contract session-kernel document; Python uses the same context_messages and
project_view functions for source cuts and context reads. Creation uses the
existing transaction create receipt. The app_server_fork attribute extension
stores lineage/default model; seed retains its closed messages/shared_state shape.
Real producer and SQL tests are in tests/test_app_server_fork.py and
tests/session/test_app_server_fork.py. No Rust adoption is required.
