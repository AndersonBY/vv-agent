# Python Contract Integration

`vv-agent` is the Python implementation of the language-neutral contract in
[`AndersonBY/vv-agent-contract`](https://github.com/AndersonBY/vv-agent-contract).
Normative behavior, fixtures, versioning, and adoption workflow live only in
that repository.

## Pinned Contract

`contract.lock.json` selects the contract version, Git revision, and artifact.
The current adoption state is not duplicated in this document. Treat
[`vv-agent-contract/support-matrix.json`](https://github.com/AndersonBY/vv-agent-contract/blob/main/support-matrix.json)
as the machine-readable source for required implementation revisions,
verification timestamp, central CI run URL, and frozen implementation baselines.
Schema 2 lists `required_implementations` explicitly and records each
implementation's own `contract_version`; schema 1 is rejected. Python is the
required implementation. `vv-agent-rs` is frozen at contract 23.0.0 / package
series 0.21.x for maintenance only and does not follow new contract releases.
A newer global contract version or Python verification run does not establish
newer Rust support. Reactivation requires a new Maker decision and full adoption
of the then-current contract through the central workflow.

The lock records the exact release artifact, artifact digest, vendored fixture
path, and canonical fixture-manifest digest. `tests/fixtures/parity/` is a
generated snapshot, not an editable source.

## Required Workflow

For any shared public, model-visible, runtime, persistence, or wire change:

1. Read this repository's lock and `../vv-agent-contract/AGENTS.md`.
2. Read the central parity, versioning, and change-workflow documents.
3. Change canonical docs and fixtures in `vv-agent-contract` first.
4. Sync required implementation snapshots with `scripts/contract_snapshot.py`.
   Preserve the frozen Rust lock and fixtures.
5. Update real Python producers and behavior tests, not only fixture parsers.
6. Run Python's full repository gates and central cross-repository CI with the
   contract and Python refs. Rust adoption and gates are not required.

Never edit a vendored parity fixture or digest directly.

## Snapshot Commands

```bash
python3 scripts/contract_snapshot.py check
python3 scripts/contract_snapshot.py check --source ../vv-agent-contract
```

After an immutable central release exists:

```bash
python3 scripts/contract_snapshot.py sync \
  --source ../vv-agent-contract \
  --artifact https://github.com/AndersonBY/vv-agent-contract/releases/download/v<version>/vv-agent-contract-<version>.zip \
  --artifact-url https://github.com/AndersonBY/vv-agent-contract/releases/download/v<version>/vv-agent-contract-<version>.zip \
  --revision <contract-revision>
```

## Workspace Edit Producer

`src/vv_agent/tools/handlers/workspace_io.py` produces the canonical edit receipt and
file metadata. The `builtin_tool_behavior_contract` producer tests consume the
central success fixture. Workspace tool tests exercise large-file consecutive
edits, stale baselines, exact replacement, and BOM/CRLF preservation.

## Verification Scope

Public producer tests establish the canonical API, runtime decisions, events,
strict wire readers, and recovery semantics under the locked contract. Passing
fixture or snapshot checks alone does not establish those behaviors.

Python producer and store tests cover the current SQLite/PostgreSQL execution
semantics. The frozen Rust v23 baseline is outside current Python adoption.

## Python producer map

| Surface | Current producer | Evidence |
| --- | --- | --- |
| Public API v8 | package exports, Runner, RunConfig, interactive and App Server | test_parity_evidence_manifests.py |
| Strict wire | events.py, types.py, message_codec.py, app_server/protocol | event validation, protocol types and session codec tests |
| Prompt and definition | runtime/compiler.py, prompt/, session/runtime.py | test_run_definition_producer.py, prompt and session fixture tests |
| Execution and recovery | session/kernel.py, session/sql.py, session/reducer.py | tests/session, App Server durable resume tests |
| Events, results and tracing | session/projection.py, session/result.py, session/tracing.py | Runner event/terminal/trace and result tests |
| Children and handoff | session/delegation.py, session/children.py | configured children, continuation, agent tools and handoff tests |
| Memory and compaction | session/compaction.py, session/memory.py, memory/ | memory lifecycle, local compaction and token usage tests |
| Budgets | budget.py, session/kernel.py | test_run_budget.py and budget/session fixtures |
| App Server v2 | app_server/, session/app_server.py | schema, protocol, turn, approval, replay and lifecycle tests |
| Workspace and tool returns | tools/, workspace/, runtime/tool_results.py | workspace parity, Bash, custom tool and metadata tests |

## Current boundaries

RunEvent v6, model-call v2, task-token-usage v3 and strict Message each accept one
current shape. Missing, stale, unknown and malformed versions reject. Results carry
session_id/turn_id. App Server
uses protocol v2 and a single ThreadStatus enum. There is no v23 selector/decoder.

The session log is execution truth. Events, results, App Server and tracing are
consumer projections; JSONL event stores are sinks. Same-turn recovery preserves
frozen prompt, model, limits and usage. Durable stores are opt-in through the
session-facing API.

The generated 45-file corpus is compared byte-for-byte with vendored v24.0.1
in one producer generation. Contract snapshot integrity is checked independently.
An implementation cannot be declared verified before required producer and central
gates establish it. See [migration-v8.md](migration-v8.md) for host migration.
