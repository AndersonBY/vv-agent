# Execution environments and workers

Status: P1 design for W4 and W3 in P3; no worker or execution-environment implementation is delivered by this document.
Authority: approved plan `vv-agent-trusted-distributed-harness-plan-2026-11-11.md` (2026-11-11 v3), sections 1.4, W1, W3, W4, W6a, W9 and 8.
Baseline: `vv-agent` 0.35.0 at `d2d02c238cc8600d30aa9af87366ccba8fedba93`; `vv-agent-contract` 38.0.0 at `1f4e6ff891996f1199f02e6b65edb1305ec7d8ae`. That repository's `support-matrix.json` at revision `13dffcfd61aa4aba800b7466295f1a7ca8675176` verifies adoption at the Python baseline. Canonical semantics are in its `docs/session-kernel.md` and record/inbox fixtures. Rust remains frozen on contract 23 / 0.21.x; its lock is unchanged.

The existing Provider protocol remains the kernel's operation boundary. ExecutionEnvironment is the resource on which a provider performs commands and file operations. A worker owns one driving connection; the environment and its commands may outlive that worker. Durable admission, SQL fencing and provider evidence keep their current meanings. S1 storage and fields are defined in [trusted-storage-s1.md](trusted-storage-s1.md).

## Callback decision

EW-D1 is settled for this design: preserve `admit(plan)`, `admit(plan,summary)` and `hook(point,record)`. The specific W4.6 rule makes an explicit transaction argument optional; it governs over section 8's conflicting shorthand that callbacks explicitly receive `tx`. That shorthand is superseded for this design, not implemented as a second API. Same-connection business/child admission remains mandatory. Any later signature change needs its own canonical change.

## Open contract decisions

| ID | Proposed choice | Required resolution |
| --- | --- | --- |
| EW-D2 | Network-capable command execution must enter the untrusted-content `network` decision path, even when its tool metadata says `side_effect="execute"`. A network-none environment may remain execute-only. | W6a's immutable untrusted-content rule names external/network, not execute. Canonical policy must approve the derived-network mapping or explicitly amend W6a. Merely relying on execute's ordinary approval default permits a host allow to bypass the threat goal. |
| EW-D3 | Allocate execution ID and stable command fingerprint before start; make remote lookup by execution ID possible even when Accepted was lost. | W3 requires recovery in that exact window but its draft `poll(handle)` signature has no pre-acceptance lookup. Include a stable pre-start handle or equivalent trusted adapter query operation in the canonical contract. |
| EW-D4 | Permit at most one logged retry of a lost remote start acknowledgment, using the ordinary supported-idempotency tool retry rule; all further recovery is query/evidence. | This makes W3's resubmission acceptance consistent with section 1.4 and I1. Transport code must not silently call start twice within the same attempt. |

## Delivery boundary

| Milestone | Required result | Not established by P1 |
| --- | --- | --- |
| P1 | W1 injection seams and seeded simulation on real SQLite/PostgreSQL; W9 import/API/doc/capability gates possible on the current code; these two design documents. Record current exceptions explicitly. | Thin local surfaces, single-owner worker connections, removal of the kernel latches and process-local background execution. |
| P2 | S1 tenant/content/chain/feature-level schema; major/level-safe admission and rolling-upgrade evidence, with reserved W3/W4 fields. | Full support-set routing, ExecutionEnvironment or the W6a threat guarantee. |
| P3 W4 | One worker driving model, compatibility filtering, independent LeaseKeeper and thin public surfaces. RunHandle becomes a cursor subscriber. | Remote environment security unless W3 is also delivered. |
| P3 W3 | Provider-backed local/container/remote environments, stable command recovery, frozen capability enforcement, secrets and egress boundary. | W7 label propagation and W6a's complete untrusted-content policy, delivered in P4. |

W9's requested convergence of Runner, Interactive, CLI and App Server on LocalWorker depends on W4. P1 can deliver import boundaries and measure existing surface exceptions; it cannot claim that convergence gate passed while those paths still drive directly. This dependency belongs in the milestone acceptance record, not behind a default-off switch. Contract 38 adoption is verified; that does not establish P1/P2/P3 feature adoption.

## Existing seams to preserve

Sources below are present at the baseline; proposed interfaces later in this document are not existing symbols.

| Source | Current responsibility |
| --- | --- |
| `src/vv_agent/session/providers.py` | `Provider.submit/query/cancel/authenticate`; `Definitive`, `Accepted`, `Unknown`; handle/result binding checks. |
| `src/vv_agent/session/runtime.py` | `Runtime.providers`, `children`, `hook`, `handler_version`; runtime rebuild; `RuntimeNotReady`. |
| `src/vv_agent/session/children.py` | `ChildTool.admit(plan)`, `SummaryChildTool.admit(plan,summary)`, child create and completion delivery on SessionTx. |
| `src/vv_agent/session/store.py` and `sql.py` | SessionTx, lease epochs, input/append receipts, scheduling and consumer acknowledgment. |
| `src/vv_agent/session/kernel.py` | Frozen operation key `{session_id}/{operation_id}`, dispatch admission, recovery and late-receipt rules. |
| `docs/host-integration.md` | Same-connection host transactions, Runtime factory, bounded retries and callback restrictions. |
| `tests/session/test_sql_host_core.py` | Django-style raw psycopg adapter and atomic business/child/parent admission test, including deferred COMMIT failure. |

This design does not depend on `execution_store()`, `tx.connection` or a connection ownership ledger. Those were not shipped in 0.35.0. No backend inspection is needed for the atomicity claim: the existing PostgreSQL adapter test supplies exact executable evidence. ECI/OpenSandbox/execd wire mapping must be inspected when implementing the host adapter; this draft makes no claim that those agents already satisfy remote deduplication or policy checks.

## Resource interface and Provider mapping

The host registers EnvironmentProvider implementations. That provider owns create, attach, inspect, pause, snapshot, restore and release. The kernel invokes attach only. A serializable descriptor contains an environment-provider ID and resource address/identity; it contains no credentials and remains encrypted as content. Worker credentials and endpoint routing come from trusted host configuration, not a model-produced descriptor. Attach rejects an unregistered provider or an unauthorized tenant/resource before issuing execution or file calls.

The environment supplies the existing WorkspaceBackend as `files`. It exposes start, poll, write_stdin, stop and search. A command's logical operation is distinct from the environment's lifecycle. Release is never inferred from a worker stopping or a session lease expiring.

| Kernel Provider call/outcome | Environment action | Required result |
| --- | --- | --- |
| `submit(plan, context)` | Attach/recheck trusted environment; call start with frozen command, operation idempotency key and expected environment binding. | Durable acceptance returns `Accepted` with a validated provider handle. A process that exited immediately may still return Accepted then be polled; a direct Definitive is allowed only with authenticated terminal evidence and full accounting. |
| `query(handle)` | Resolve the original execution, call poll, read output by stable cursor. | Running returns Accepted with the same execution identity; terminal returns Definitive ToolExecutionResult; absent/unprovable outcome returns Unknown. Never substitute start for poll. |
| `cancel(handle)` | Call stop on that execution. | Stop acknowledgment alone returns Accepted or Unknown; return Definitive only after trusted terminal state confirms outcome. Already exited returns its original terminal result. |
| `authenticate(item, plan)` | Verify trusted transport/provider evidence, actor, operation/attempt/request, provider, target turn/generation, execution/environment identity, result and usage. | A claimed job ID, model-produced result, digest, descriptor or evidence string is insufficient. Authentication precedes input application. |
| `Definitive` | Exit status and retained output/result from the original execution. | Strict ToolExecutionResult with matching call ID, valid status, nonempty evidence and original acceptance evidence when parked. Result/usage/shared-state validation and duplicate conflict rules remain in force. |
| `Unknown` | No trustworthy completion or acceptance can be recovered. | Preserve evidence and uncertainty. Never map transport timeout, missing PID, unavailable output or stop request to successful completion. |

`bash` owns the command-start operation and exposes the stable execution reference in its tool result when backgrounded. `check_background_command` and `stop_background_command` are later operations targeting that original reference; they cannot retarget by PID or start a replacement. File tools (`read_file,write_file,edit_file,find_files,search_files,file_info`) and search use the attached environment and existing file authorization. A file mutation is still an operation subject to its own declared idempotency; attaching a remote filesystem does not make write/edit automatically safe to retry.

`write_stdin` changes process state. Do not replay an unacknowledged stdin write automatically. If a future surface exposes retryable stdin, the executing agent must deduplicate a stable per-write sequence/operation ID first; W3 does not create an extra reliable input protocol for an unused surface.

## Frozen environment binding

The trusted EnvironmentProvider returns this closed capability declaration:

```text
EnvironmentCapabilities = {
  environment_id, generation,
  isolation: none | container | microvm,
  network: open | allowlist | none,
  enforces_network_policy: bool,
  enforces_resource_limits: bool,
  persistent: bool,
  network_policy_digest, isolation_policy_digest,
  network_policy, isolation_policy, evidence
}
```

`generation` is a nonnegative monotonic integer for that resource. Every change to effective isolation/network/resource policy or restored incarnation advances it; IDs/generations must not be reused after resource recreation.

This is the runtime/operational declaration, not the clear ledger shape. Operational `network_policy_digest` and `isolation_policy_digest` are ordinary `SHA256(JCS(policy_document))` over complete effective policies, including allowlisted host:port pairs and resource limits. The trusted provider also supplies `network_policy` and `isolation_policy` objects `{document, operational_digest}` and verifies the hashes against those documents. The remote agent recomputes these ordinary hashes for generation/start equality; it needs no session encryption or HMAC key.

Persist the policy objects and expected operational tuple only as encrypted S1 content. Clear ledger fields `network_policy_ref_digest` and `isolation_policy_ref_digest` instead hash the committed reference forms of those policy objects, as defined in the storage design. No naked private operational policy hash belongs in clear SQL, skeleton records/exports, derived IDs or ordinary logs. Evidence is also encrypted audit content; the host provider verifies third-party attestations if needed. The kernel does not introduce cryptographic remote attestation. Exact new policy-slot/field allocations remain canonical work before implementation.

At turn admission, retain descriptor, capabilities, their reference-form digest and the ledger-domain policy digest in turn_started. Derive worker requirements from the same frozen definition in that transaction. Record each command's environment ID/generation, policy references and ledger reference digests, and stable command digest with its operation. The expected operational tuple `{environment_id,generation,network_policy_digest,isolation_policy_digest}` remains encrypted. Hydrate it for attach/start equality; ledger reference digests cannot substitute for operational hashes. Retries and recovery use that original binding, not current default environment selection.

Attach always obtains fresh capabilities from trusted host code. The kernel compares resource identity and generation/digests with the frozen tuple and verifies required enforcement flags. An isolation downgrade, wider network or lost persistence guarantee is a deterministic denial before new effects; do not quietly substitute LocalEnvironment. For an active turn the tuple must match exactly, even if a new policy appears stronger. A new generation belongs to an explicitly admitted later turn; equality avoids ambiguous partial ordering of allowlists. Poll/cancel of an already accepted command may still use its historical execution identity to reconcile or stop it; a policy mismatch is not evidence that the old effect failed.

Every start carries the hydrated expected operational tuple over the authenticated transport, not the ledger reference digests. The remote agent atomically checks it while accepting the command and immediately before spawning under that policy. It must bind the actual sandbox/process policy to that accepted generation so that a concurrent policy update cannot release a process under a weaker policy. Mismatch returns authenticated `not_started` rejection with no process. Checking only at attach or only at HTTP request parsing is insufficient.

If policy changes while a command is already running, start-time validation cannot retroactively undo its effects. The EnvironmentProvider must preserve its accepted restrictions or stop/revoke the command when the policy can no longer be enforced; changing effective policy advances generation. Poll must report the changed binding, and the kernel must not authorize new effects using the old assertion. Evidence of changed policy is recorded without fabricating a terminal outcome.

LocalEnvironment wraps current process and reaper code, declares isolation none, network open, no network-policy enforcement and persistent false. ContainerEnvironment advertises only what its Docker/Podman setup enforces. RemoteEnvironment advertises persistent true only if commands, outputs, workspace and command identity survive worker replacement; survival of the execution agent itself is a separate tested retention property. A lost execution-agent registry yields Unknown, never PID-based adoption.

Only LocalEnvironment modules may use subprocess after W3 conversion. Read/search tools must not keep helper Popen paths in the worker. The existing WorkspaceBackend API and artifact ACL checks remain, with file I/O executed at the environment boundary.

## Stable command identity and remote acceptance

For each command, freeze these values before dispatch:

| Value | Lifetime |
| --- | --- |
| `tenant_id,session_id,operation_id` | Logical operation identity. Tenant is authenticated transport scope, never inferred from the operation key. |
| `idempotency_key` | Existing literal `{session_id}/{operation_id}`; unchanged across attempts and workers. Server deduplication key is `(tenant_id,environment_id,idempotency_key)`. |
| `execution_id` | Opaque deterministic ID allocated before first start, derived from a domain-framed hash of authenticated tenant, environment ID and stable operation key. It is not a credential. Retained in the plan and reconstructible before Accepted exists. |
| `command_digest` | Session-keyed digest of canonical CommandSpec plus frozen environment/policy tuple. Excludes per-attempt request/call/dispatch IDs, clocks and transport credentials. Includes command text, cwd, stdin/argv choices, secret-reference names, limits and effective network policy. |
| `attempt,request_digest,dispatch_id` | Ledger attempt identity; changes only on an allowed logged retry. Distinct from the stable command fingerprint. |

Use HMAC for the command fingerprint as for other content-derived evidence in S1; the remote agent compares the opaque fingerprint and validates the actual immutable command associated with it, without receiving the session HMAC key. The fingerprint must not expose a plaintext command hash in the ledger. Bind the full expected operational policy tuple separately in the authenticated request; keep its private operational hashes inside encrypted content when persisting it in S1.

The remote acceptance store durably maps the deduplication key to execution ID, command fingerprint, policy tuple and status before returning acceptance. A duplicate key with identical immutable inputs returns the original execution, running or exited. A duplicate key with changed inputs is a deterministic conflict and spawns nothing. Deduplication state and terminal output remain available for the session's reconciliation lifetime; expiry cannot make an old key usable again. Retain a spent-key tombstone after output GC. Resource recreation uses a new environment ID/generation and cannot silently revive a frozen operation.

The agent must close its own accept/spawn crash window. It needs an execution-ID-addressable process supervisor or an atomic create-if-absent execution resource so that replaying a reserved acceptance cannot create a second process. Persisting a row before Popen, without a way to find that exact process after an agent crash, is insufficient. If the agent cannot determine whether the reserved process started, report Unknown and refuse a new spawn for that key. W3 requires worker-crash recovery and at-most-one process; it does not promise recovery of an irretrievably lost agent registry.

For a lost Accepted response, recovery first queries using the pre-start execution identity. A definitive absent/not-started response can allow the one permitted retry; unavailable lookup stays uncertain. If resubmission is needed, record `op_unknown` and a new attempt with supported idempotency before its submit. The same key returns the original process even if it started in the first attempt. Once two attempts have been submitted, query/evidence/manual reconciliation are the only remaining routes. No hidden HTTP retries issue start again under the same ledger attempt. Poll transport retries are harmless only when they query, never submit.

A returned existing execution can belong to an earlier attempt. The trusted adapter issues acceptance evidence bound to the current attempt/request and the same underlying execution, retaining the original acceptance provenance. It does not rewrite old handles or claim an additional process. Completion of that logical execution must not double-count its resource cost across retry/adoption/audit receipts; transport-attempt usage is recorded separately if it is actually chargeable. Unknown costs remain unavailable, never zero. The kernel's existing model-attempt billing rule remains unchanged.

## Recovery cut matrix

Before treating a started operation as unknown, recheck committed receipts and the inbox watermark for authentic acceptance/result evidence. Receipt replay grants no lease or dispatch authority.

| Cut / retained evidence | Recovery action | Prohibited action | Invariants |
| --- | --- | --- | --- |
| Planned/prepared committed; no started record | Reuse frozen hook result, recheck authorization/budget/binding, admit one dispatch under a live lease. | Treat a plan as proof work already happened or rerun a committed preparation hook. | I1, I2, I8 |
| Started committed; worker dies before actual send | Ledger cannot distinguish this from an uncertain send. Apply the operation-specific unknown rule below. Remote query may prove not-started. | Assume no effect solely because a response is absent. | I1, I10 |
| Ordinary model sent; no receipt | Record unknown, duplicate-cost risk and missing measurement; retry as a new logged attempt up to `max(2,frozen endpoint count)`. | Reuse attempt ID or zero missing usage. Unknown output_repair is never retried. | I1, I5, I10 |
| Ordinary supported-idempotency tool sent; no receipt/handle | At most one logged retry, same operation key; caller actually deduplicates. | Declare idempotency from metadata if the provider cannot recover that key. | I1, I5, I10 |
| Non-idempotent unqueryable tool sent; no receipt | Keep unknown pending trusted evidence/manual resolution. | Automatically resubmit or declare failure/success. | I1, I10 |
| Queryable provider accepted; handle parked | Query the original handle, reuse running acceptance or authentic terminal result. | Resubmit because poll timed out or a worker changed. | I1, I5, I10 |
| Remote start accepted/process created; Accepted response or SQL park lost | Query pre-start execution ID; if needed use the single logged supported retry with the same key. Recover original execution and acceptance. | Start another process, retry submit under the same attempt, or invent a handle from a PID. | I1, I2, I10 |
| Remote acceptance persisted but agent dies around spawn | Supervisor/agent reconciles the stable execution identity; otherwise report Unknown and do not spawn again. | Reset deduplication state on agent restart. | I1, I10 |
| Accepted/parked commit succeeds but reply is lost | Reload/replay commit, query original execution. | Treat lost SQL reply as permission to start again. | I1, I3, I8 |
| Command exits; terminal output not yet in SQL | Query same execution, validate evidence/result/usage and commit once. | Rerun command to recover output; map missing output to empty successful result. | I1, I5, I10 |
| Model/tool result committed; worker dies before return/projection | Reuse receipt; reconstruct missing dependent plans from that result; continue consumer cursor. | Call the model/tool again or reserve/bill the same receipt twice. | I1, I5, I8 |
| Local worker/process manager disappears | Only original owner/manager evidence may reconcile; otherwise Unknown. | Reattach by PID, start a replacement, claim remote-style persistence. | I1, I10 |
| Cancel/stop sent; outcome unconfirmed | Retain acceptance and unconfirmed operation, poll/authenticate later; terminal state must not revive. | Turn stop request acknowledgment into confirmed termination. | I7, I10 |
| Trusted late receipt after retry/turn/generation changes | Apply existing normal/correction/audit rules and account real usage once. Cancelled/ended turns stay ended. | Overwrite consumed unknown context or revive an old turn. | I3, I5, I8, I10 |
| Lease expires/takeover during call | SQL rejects stale append; successor recovers via above rows. External admitted zombie request may still arrive; remote key deduplicates it. | Claim SKIP LOCKED or epoch checks cancel an already sent request. | I1, I2 |
| Callback/append/COMMIT fails | Roll back business and kernel admission together; retry from the committed prefix using retained operation/input/commit identities; reuse any already staged content. | Commit host rows independently or report a provisional receipt as committed. | I2, I3, I6, I8 |

A command rejected before spawn by the agent's generation/policy check is a deterministic non-execution result. A policy mismatch discovered after acceptance is not proof of non-execution. Transport failure while delivering a rejection remains uncertain until that signed/trusted rejection can be recovered.

## Worker supports and session requirements

Use these immutable process-local declarations and durable requirement values:

```text
WorkerSupports = {
  storage_major: "S1",
  max_level: positive integer,
  handler_versions: set[opaque handler ID],
  capabilities: set[opaque capability ID]
}
SessionRequirements = {
  level: positive integer,
  handler_version: opaque handler ID | null before first turn,
  capabilities: sorted unique opaque capability IDs
}
```

Example capabilities are `env:remote`, `toolset:<digest>` and host adapter version IDs. These are exact membership labels; there is no wildcard, numeric ordering of handler IDs or live registry. Capability IDs cannot contain content or credentials. A tag asserts a worker can reconstruct the frozen implementation; it is not proof that an environment enforces policy. Environment freshness is checked separately.

A trusted host supplies bootstrap requirements at session creation, including a capability identifying the runtime factory/agent family. This lets a worker filter before it needs an uncompiled turn. A null pre-turn handler does not mean any arbitrary worker can execute: bootstrap capability containment still applies. First turn admission derives exact handler/capability requirements from the actual frozen definition and stores them with turn_started in the same transaction. If compilation finds a requirement outside this worker's support set, do not dispatch or fail the session; leave compatible work for another worker and record the routing mismatch operationally. Bootstrap declarations must be corrected by trusted host administration before repeated routing; do not broaden them automatically.

For an active turn, handler/capability requirements remain frozen. Level is the monotonic highest read requirement for the session, including inbox and dependency records; it cannot fall at the next turn. A later turn can select a different exact handler or capability set through a new frozen definition, but must use trusted pre-admission requirements so the wrong worker cannot consume its input. Content references and provider evidence do not get to advertise their own authority.

The frozen runtime definition must also bind counter/tokenizer model identity, counting method/implementation, asset identity and normalization, with compatible-worker requirements reserved at the sole S1 boundary; exact canonical allocation remains a P2 decision (see [trusted-storage-s1.md](trusted-storage-s1.md#reserved-fields-and-activation)). Current vv-llm's shared `"qwen"` cache makes the selected tokenizer object depend on initial model order. P1 preserves that behavior and makes no cross-restart determinism claim. Compatibility evidence must initialize differing model orders and restart workers, then prove the frozen binding/counts are reproduced or the worker is excluded before replay/dispatch. Do not silently choose a different tokenizer, duplicate defaults, add historical decoders or change vv-llm during extraction.

The execution claim predicate is:

```text
storage_major == supports.storage_major
and deletion_state == live
and session.level <= supports.max_level
and (handler_version is null or handler_version in supports.handler_versions)
and required_capabilities subset_of supports.capabilities
and drive_retry_at_ms <= database_now
and no unexpired execution lease
and (next_drive_ms <= database_now or ready unconsumed input exists)
```

PostgreSQL applies it in the candidate query before `FOR UPDATE SKIP LOCKED`, rechecks it under the selected session row lock, then increments epoch and assigns owner/until in the same claim transaction. SQLite uses the identical predicate inside its serialized write transaction. A direct wake, explicit session-ID acquire or resume must use the same filter; no alternate acquire can bypass compatibility. Scalar levels and handler IDs can be SQL columns; capability arrays use the S1 canonical representation with adapter-specific containment implemented without decoding log bodies. P2 may select candidates by indexed columns then compare the small requirement array under lock, but must skip without acquiring a lease on mismatch.

A new high-level input or new requirements update locks the session row and raises/updates its cached requirements in the same transaction as publication. An already leased old worker must recheck current requirements before decoding newly admitted data and before dispatch/append. If it no longer qualifies, it stops safely and releases; this is not a definition-drift terminal. Level checks include prefix/projection dependencies, so an old parent driver cannot decode a new child inadvertently. Projection consumers need supported S1/level filtering too, even though they do not take execution leases.

After claim, rebuild with trusted `build_runtime`. Classify failures precisely:

| Condition | Action |
| --- | --- |
| Requirements exceed this worker's declared support | Skip before claim; if requirements changed after claim, release before reading/dispatch. No terminal. |
| Resource/model/key preparation temporarily unavailable (`RuntimeNotReady`, `KeyUnavailable`) | Set bounded retry gate, release, preserve inputs/turn/waits. |
| Worker explicitly declared support but reconstructed frozen definition digest differs | Existing deterministic definition-mismatch closure; preserve unconfirmed external work. This is not ordinary heterogeneous routing. |
| Environment policy/generation mismatches frozen command binding | Deterministic denial of new execution; retain/reconcile original accepted work. |
| No worker supports a due session | Keep pending/due; count unmatched work and age through host metrics, alert when threshold is crossed. W8 owns standardized telemetry. Never fabricate a turn failure or spin a tight retry loop. |

`list_runnable` and wake remain hints; only the compatible claim plus current epoch grants driving authority. A failed wake does not remove SQL readiness. Projections keep independent cursors and may not wait under a lock held by a driving worker.

## W4 transaction and connection contract

One worker driving thread owns one SQL connection/store while driving. Local surfaces own separate submission/read connections; LeaseKeeper owns a separate connection and updates only lease state. The timeout invoker may retain its execution thread but must never receive a store connection. No connection ownership ledger, cross-thread borrowed raw connection or memory-store exception is introduced.

Local default storage becomes a temporary SQLite WAL file whose owner controls cleanup after workers/subscribers close. Remove the :memory: role in W4. Runner, Interactive, CLI and App Server write inputs in short transactions, wake after commit and subscribe/read by cursor. RunHandle uses those same read/input paths; it owns no parallel driver. No surface waits for a worker while holding a lock the worker needs.

A host transaction can atomically write a product input/business row and inbox input, then wake after outer commit. It cannot enclose drive, runtime rebuild or provider I/O. A consumer transaction can atomically project host business/billing rows and acknowledge the batch. External sinks use stable IDs and cannot pretend to join SQL.

Child admission and transactional hooks execute in the worker-owned transaction, using the exact host adapter connection. Under the settled EW-D1 decision, signatures remain `admit(plan)`, `admit(plan,summary)` and `hook(point,record)`; callback closure/binding receives the host adapter that wraps the active driving connection. The adapter must integrate the matching ORM transaction. Do not bind a request-thread connection or open a fresh connection for its business writes. Post-commit notification hooks remain post-commit; only hooks documented as transactional may perform atomic admission writes.

The atomic admission unit includes callback business rows, every child's creation/initial inbox input, parent started/parked/background result, budget reservation, content links and commit receipt. Empty/malformed/mixed child batches reject as today. A failure at validation, second-child creation, head/inbox CAS, lease fence or outer deferred COMMIT rolls back the entire unit. Wake/result notification is after actual outer COMMIT. Returning from append inside a savepoint is not durable acceptance.

Host callbacks perform short database checks/writes only. File copies, downloads, summary generation and other arbitrary callback I/O stay outside admission; no irreversible business API calls, independent business commits or child execution loops are allowed. Stage known key/content inputs outside the transaction when possible and recheck any staged identity/receipt during admission.

S1-D7 permits narrowly scoped SDK key allocation/unwrap, encrypted content persistence and checkpoint signing inside the existing admission transaction when the child IDs/content become known only after the callback runs. This is SDK persistence, not a new permission for host callback effects. Preserve callback signatures and the original atomic unit; no mandatory child discovery/rollback/prepare/rerun API or separate preparation ledger is required. Any uncommitted external allocations/uploads are scoped orphan cleanup, not proof that business admission committed.

All SDK requests/retries share a cumulative admission deadline within the last confirmed lease's remaining safety budget, reserving time for final validation and outer COMMIT. Bound network work explicitly; SQL timeouts alone do not bound it. After persistence and signing, recheck owner/epoch, database-time expiry, head/inbox CAS, deletion fence and current requirements/write ceiling. Insufficient budget, unavailable key/content service, timeout or failed checks roll back every business/child/parent SQL write; defer/retry from the committed prefix under existing identities. Late external completion remains subject to orphan cleanup and publication/deletion fences. Do not run provider effects or launch children until admission has actually committed.

Callback invocation remains at least once when a crash precedes commit. Existing input/operation/commit receipts handle replay and lost commit replies; reuse any already staged content when available. At implementation, explicitly qualify `docs/host-integration.md`'s file/network-I/O restriction for bounded SDK persistence only, retaining its prohibition for arbitrary host callback effects.

LeaseKeeper renews independently without the current kernel's commit/poll/state latches. Commit safety comes from checking owner, epoch, current database time, head and inbox CAS inside the append transaction, including a final expiry check. Lock waits and statements are bounded by remaining lease time. Renewal can be blocked by an admission row lock; callbacks and SDK persistence must fit the admission safety budget or their whole SQL unit rolls back. Renewal failure never grants an extension and cannot rescue an expired commit.

For L5, proposed W4 classification is: SQLite BUSY/LOCKED on renewal is a retryable contention error only while the last confirmed lease retains the configured safety margin; expiry/insufficient margin is LeaseLost. Owner/epoch mismatch is immediately LeaseLost. Never infer a successful renewal from a busy response. Retain the latest confirmed LeasePoll's lease, control items and database time together during handoff (L2). Controls bind to their original turn/generation/token (L3). Custom stores without database time must use a pre-acquisition monotonic timing bound or be rejected for W4 lease-driving support; do not construct authority from a post-acquisition wall-clock sample (L4).

Reuse `tests/session/test_sql_host_core.py::test_child_admission_business_and_session_increment_is_atomic` and its HostPostgresStore adapter. Its current cuts include invalid member, second child, CAS, lease and deferred outer COMMIT. Extend it through the W4 public worker path with both prestaged content and bounded SDK persistence for children discovered by admission; use a second connection to prove committed business/child visibility. Preserve `test_public_drive_and_dispatch_tick_preserve_host_instance` behavior for host subclasses. No new backend replica or general connection-management subsystem is needed.

## Policy, secrets and trust boundary

Environment policy is supplied by trusted host code and frozen policy decisions, never model arguments. A tool requesting a wider network cannot override it. If required network/resource enforcement is unavailable, reject before external work. LocalEnvironment cannot satisfy network none/allowlist enforcement. A process must not receive a more permissive workspace mount or filesystem backend than the attached approved environment.

W6a first defines ordinary strict-mode defaults for external/network/execute, which a trusted host may explicitly allow. Its separate rule for untrusted context mandates approval for external/network and forbids host bypass. Under EW-D2, execute with open or allowlist network contributes a network capability to that non-bypassable rule; scanning shell text for curl is not sufficient. Environment network none can be execute-only only when the environment enforces it. Side effects mediated by an injected credential/proxy also need classification; an unrestricted broker or mounted host service cannot bypass the network decision. This mapping must be resolved before claiming the plan's injection/exfiltration defense; P3 alone cannot claim it before W7/W6a.

Approval remains bound to the exact prepared request/attempt, request digest, turn/generation, scope, policy digest, approver and expiry. A command/environment binding change invalidates its approval. Frozen denies remain denies; `can_use_tool` can only tighten later policy. A model endpoint receives principal for routing/audit outside the prompt; secrets and Principal attributes do not enter model context merely because the worker can decrypt them.

The 2026-11-12 P3 mechanism revision preserves plan section 3: arbitrary command code must not obtain plaintext keys. SecretBroker and the existing trusted environment/operation providers retain raw credentials in trusted broker/provider code or host adapters. A `secret://name` request is not itself authorization. Untrusted code receives only an opaque reference bound to tenant/principal/session, environment ID/generation, operation and permitted service/action, or access to the adapter's mediated authenticated operation. Each use checks that binding; the interface exposes neither raw-secret retrieval nor arbitrary credentialed requests. Raw-key injection is permitted only into trusted host adapters, never model-authored arbitrary command processes or their readable environment, files or mounts; command children must not inherit adapter credentials. Unsupported credential isolation or mediation rejects before command start/use, with no raw-key fallback. Reuse these provider/broker boundaries; P3 does not require a generic HTTP proxy, 2PC or another secret framework. This implementation revision requires canonical review and does not claim new explicit Maker approval.

Sensitive-value redaction in output, streaming, spool, diagnostics and remote logs remains supplemental, including chunk boundaries. It is not the isolation boundary: a program that can read a key can encode it before output. Secret-reference names and arbitrary command text remain encrypted content.

## Simulation invariants and recovery promises

Simulation uses the real SQLite file and PostgreSQL stores, one connection per participant. FaultyStore wraps cuts; it is not a replacement storage model. Execution agents/model/providers are deterministic scripted participants, with durable command-key state shared across worker replacements. Each failing trace includes seed, code revision and contract version. Conditional liveness requires fair scheduling, restored services and at least one compatible worker; all-incompatible or unavailable-service worlds remain pending and observable.

| Invariant | Recovery promise / required oracle |
| --- | --- |
| I1: one external invocation per operation/attempt | Count actual submit sends, not only started records. New sends need a legal new attempt and the unchanged operation key. Separately count physical remote process creations: at most one per deduplication key, including lost Accepted, delayed stale worker and agent accept/spawn cuts. Model physical SDK retries remain governed by their existing observation contract. |
| I2: no stale-epoch writes | Expire/take over between admission, renewal, callback and append. Stale appends and business/child writes roll back; SKIP LOCKED alone never satisfies this invariant. |
| I3: input applied once or pending | Duplicate/reordered controls, authenticated callbacks and commit-reply loss preserve one input disposition. Test both direct inbox and embedded input copies plus tenant/actor bindings. |
| I4: no lost wake at rest | Drop/duplicate/delay wakes before and after release; tick finds due work. A terminal unprojected record remains consumer work. With compatible available workers the world cannot be idle while ready active work has no pending scan/wait. Treat all-incompatible sessions under I11, not as successful execution. |
| I5: usage once per attempt | Committed model/tool usage and late audit attempts appear once. Unknown measurement remains unavailable; retry duplicate-cost risk stays visible. One deduplicated command execution cannot create duplicate resource charges just because two transport attempts recover it. |
| I6: authentic child delivery at most once | Commit parent/child admission together; authenticate against the original child and delivery target. Duplicate completion push+ack or rollback cannot create a second adopted child result. |
| I7: cancellation reaches descendants | Under fair recovery and a provider capable of confirming stop, all descendants terminate. Unqueryable/unconfirmed external effects stay listed as unknown/unconfirmed; the oracle never invents remote termination to satisfy liveness. |
| I8: prefix replay equals incremental fold | Replay every committed prefix including frozen requirements/environment decisions, compacted message references and recovered handles. Scheduler indexes and content hydration cannot become another execution ledger. |
| I9: valid digest chain | Activate after W2c. Corrupt/delete/reorder prefixes and check reference-form digests/checkpoints. Whole trailing checkpoint deletion requires a trusted external head; do not make the simulator assert stronger database-only assurance. |
| I10: unknown changes only by evidence/permitted retry | Enumerate model, supported tool, ordinary tool, queryable provider, remote and local-process rows of the cut matrix. Wrong actor/provider/attempt/generation/policy evidence cannot settle or revive. |
| I11: incompatible worker never claims | Mix major/level/handler/capability support and mutate requirements through inbox admission. No mismatch gets a lease/body decode. With no match keep pending, emit observable unmatched age, and resume when a compatible worker is added. |

L2–L4 require a baseline failing seed or a precise explanation why no invariant fails, as the plan requires; do not turn unproven robustness concerns into established defects. L5 is a policy/classification choice, with contention tests after resolution. The current L1 lock-order mutation remains part of P1 evidence, while W4 eventually removes those latches altogether.

## Acceptance cases

| ID | Required evidence |
| --- | --- |
| EW-A1 | Provider mapping uses real kernel submit/query/cancel/authenticate with strict results and usage. Running, terminal, invalid, missing and cancellation-ack-only responses preserve their exact meanings. |
| EW-A2 | Start a persistent remote background command, kill worker, attach on a different worker, read output and stop the same execution. Repeat with worker death after remote start but before Accepted is committed. One process, one stable key, legal attempt count. |
| EW-A3 | Duplicate key with equal command returns original handle/terminal; altered command or environment binding rejects. Agent restart at reservation/spawn/response cuts cannot create another process. Old keys stay spent after output retention expires. |
| EW-A4 | Weaker capabilities at attach reject. Change remote network policy after attach and before start: agent rejects before spawn. Race start with policy update and verify no execution under weaker policy. Test a remote agent without session keys: ordinary operational hashes still match, while ledger reference digests are not accepted as operational hashes and private hashes never enter the clear ledger. Reattach to a new incarnation fails for the old frozen tuple. |
| EW-A5 | Local manager loss yields Unknown without PID guessing/relaunch. A cancelled remote operation with unconfirmed stop remains unconfirmed and accepts only audit evidence after turn end. |
| EW-A6 | Claim filtering covers scan, direct wake/acquire/resume, active old lease with new high-level input, child read dependencies and projection. No incompatible worker decodes body or calls runtime/provider. All-incompatible work stays pending until a compatible worker joins. |
| EW-A7 | PostgreSQL same-transaction adapter test through public worker path preserves callback signatures and includes every existing failure cut, prestaged content and callback-discovered children using bounded SDK persistence. Inject key/upload/signing timeout and expiry after that work; business rows, all children, parent admission and reservations commit or disappear together, proved by an independent TCP/password connection. Only cleanup-owned external orphans may remain. |
| EW-A8 | Public surfaces use independent connections and temporary SQLite WAL. Concurrent cancel/steer/read/reconnect cannot deadlock a driver; subscriptions replay via cursor. Static checks prove removal of old driver/pump/latch code and subprocess confined to LocalEnvironment. This is P3 evidence, not P1. |
| EW-A9 | Network none rejects LocalEnvironment and blocks container curl. With untrusted context, a host-allowed execute command in a network-capable environment cannot bypass mandatory network approval under the approved EW-D2 resolution. Test allowlist exfiltration too. |
| EW-A10 | Arbitrary command code reads its environment and accessible files/mounts but cannot obtain the canary raw key; authorized mediated service use succeeds. The same reference fails in another principal/tenant/session, environment/generation or operation, and for an unauthorized service/action. Unsupported isolation/mediation rejects with no raw-key fallback or credential inheritance. Test supplemental split-chunk redaction across DB, stream, spool and worker/agent logs; absence of literal key output alone does not prove isolation. |
| EW-A11 | Seeded SQLite/PostgreSQL traces cover every recovery cut and I1–I11. P1 required mutation detections remain within 1,000 seeds per mutation/store; P3 million-seed run and historical seed regressions pass remotely. Report seeds and revisions, not unsourced exactly-once claims. |

These are implementation acceptance requirements. P1 document completion does not assert that any new W3/W4 runtime gate has passed.
