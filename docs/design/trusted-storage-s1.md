# Trusted storage S1

Status: P1 design for P2 (W2a–W2d); not an implemented storage contract.
Authority: approved plan `vv-agent-trusted-distributed-harness-plan-2026-11-11.md` (2026-11-11 v3), sections 1.2, 1.4, 3, W2, 7 and 12. V3 and the six Maker decisions remain approved. The 2026-11-12 implementation revision below repairs the integrity mechanism while preserving those goals; it requires review and canonical acceptance before P2, and does not claim new explicit Maker approval.
Baseline: `vv-agent` 0.35.0 at `d2d02c238cc8600d30aa9af87366ccba8fedba93`; `vv-agent-contract` 38.0.0 at `1f4e6ff891996f1199f02e6b65edb1305ec7d8ae`. Canonical inputs are that repository's `fixtures/session_record.schema.json`, `fixtures/session_inbox.schema.json` and `docs/session-kernel.md`. Adoption is verified for this Python baseline in `support-matrix.json` at contract-repository revision `13dffcfd61aa4aba800b7466295f1a7ca8675176`; Rust remains frozen at contract 23 / 0.21.x.

S1 replaces the current storage format once. It separates replayable control facts from encrypted content, scopes every session datum by tenant, and chains the stored reference form. Existing recovery, input deduplication, receipt identity and atomic child admission remain mandatory. The execution and claim rules are specified in [execution-worker-contract.md](execution-worker-contract.md).

## Design decisions

Small content is inline only as ciphertext. No user, model or host plaintext belongs in the skeleton. P2 uses the approved disposable .ai drain exception; .com first adopts the kernel after P2. Signed deployments require per-record signatures from creation and admission signatures for inputs, both through LedgerSigner. Suffix completeness still needs a trusted independent expected head. Database tenant isolation binds authenticated tenant logins; SET LOCAL alone is not an identity boundary.

| ID | Decision | Basis |
| --- | --- | --- |
| S1-D1 | Inline only encrypted content. No plaintext content inlining, including values below 1 KB. Small ciphertext may be stored in SQL. | W2d.2 permits small inline values, but W2d.6 and its plaintext scan require erasure and no content in the skeleton. Treat the threshold as a placement optimization for ciphertext only. |
| S1-D6 | P2 delivers the versioned S1 export format, but no migration tool for disposable pre-S1 .ai sessions. | W2a's general major-release migration requirement is qualified by the explicit P2 exception in sections 7 and 12. Future major releases still require migration coverage. |
| S1-D7 | Preserve atomic business/child admission. Stage known content outside the transaction when possible; permit narrowly scoped, bounded SDK key/content/signing work inside it when child content is discovered during admission. No mandatory multiphase child preparation API. | Final lease/CAS and deletion-fence revalidation follows persistence work; timeout or failure rolls back the whole SQL admission. External orphan allocations are cleaned separately. Arbitrary host callback effects remain forbidden. |

## Open contract decisions

These proposals require canonical acceptance before P2 implementation. They do not alter mandatory recovery, privacy or transaction semantics silently.

| ID | Proposed choice | Reason / required resolution |
| --- | --- | --- |
| S1-D2 | Content-dependent ledger digests use the S1 reference form; command fingerprints use a session-keyed HMAC. Operational policy digests use ordinary canonical hashing for remote equality but reside in encrypted content when persisted in the ledger. | The digest-domain choice is settled for this design; canonical field names, schemas and fixtures still require acceptance at P2. Do not retain naked private-policy/request/result/definition/source hashes in skeleton fields or derived IDs. |
| S1-D3 | `KeyProvider.destroy(tenant_id, key_id)` revokes an independently destroyable session key capability, including restored wrapped copies. | A wrapped DEK in SQL under a long-lived shared KEK does not meet this guarantee. The local provider needs an external session-key registry; the Aliyun adapter needs equivalent per-session revocation. The deployment must prove this property before claiming crypto-erasure. |
| S1-D4 | The durable discriminator is the string `"S1"` in both logical envelopes and SQL `schema_version`; SQLite `user_version=4`. Initial feature level is 1. | The plan names S1 but does not assign wire/SQL values. Exact values need canonical allocation, not reuse of the current record 1 / inbox 3 / SQL 1 boundaries. |
| S1-D5 | `ContentLabels.source` is a sorted, duplicate-free array of source kinds, not a scalar. | W7 requires source-set union; its single-source examples do not define the aggregate wire. Empty source set is the P2 inactive default, never evidence of trust. |

## Format and types

`S` below means a clear skeleton field. `C` means the entire value, recursively, resides in encrypted content; the stored field contains a typed content reference instead. `S/C` means a closed object whose children are partitioned explicitly below. Null remains null; absent optional fields remain absent. Empty non-null content values also use references. Arbitrary JSON is never recursively guessed to be safe because some leaves are numbers.

The logical S1 record and inbox contain the reference form. Hydration produces runtime objects in memory; it is not a historical decoder. Only strict S1 readers run in the S1 pool. Normal execution requires available content. Ledger verification requires only skeleton bytes and public signing keys, so it remains possible after forgetting content.

`ContentRef` is closed: `{content_id, mac, size, key_id, labels}`. `content_id` is an opaque immutable identifier allocated by the content store; it is not an S3 URL, filesystem path or plaintext hash. `mac` is lowercase HMAC-SHA-256 hex of plaintext using the session content key. `size` is plaintext byte length. `key_id` is an opaque session-key identity, not a provider credential. `labels` is closed: `{source: [system|user|internal|untrusted], classification: public|internal|confidential|restricted}`. All five fields are S. Source order is the listed order; classification has the listed increasing order.

Every C field has a statically specified content slot in its kind's schema; a user object resembling `ContentRef` cannot be interpreted as a control reference. JSON content uses RFC 8785 bytes; binary content uses exact bytes and a schema-defined media type. Readback decrypts, validates size and MAC, then validates the original typed content boundary. AEAD or MAC mismatch is integrity failure. Missing content is `ContentUnavailable`; unavailable key service is retryable `KeyUnavailable`. Neither becomes a fabricated tool/model result.

Messages are separate objects. `request.messages`, creation seed messages, hook snapshots, compaction replacement and retained replies contain ordered arrays of message references. A repeat of one message reuses its reference within that session and label set. The rest of a model request is an encrypted object referring to that message array. Do not encrypt every full conversation as one new blob at every step. Generic opaque content containing known message arrays must use the producer's typed extraction path; unrelated arbitrary JSON remains one C object.

Before encryption, that typed extraction enumerates every content reference needed to hydrate the owning record/input, including message references inside encrypted containers. Publication flattens them into `sk_content_link` under the record/input owner and schema-owned slot paths with array positions, such as `request.messages[0]`. Link both the outer request object and every nested message; linking only the outer ciphertext is insufficient. Repeated message references may occupy several slots while sharing one content object. Reuse the same references when consuming an inbox item and publish its record-owned links atomically. GC needs no decryption or general content graph to protect these objects.

Identifiers retained as S must be opaque stable IDs, validated kernel enums or registered binding IDs. They must not embed prompts, customer names, paths, URLs, tokens or free-form exception messages. Tool names are the plan's explicit exception. Hosts that currently use business text as an ID must allocate an opaque ID before crossing S1. Sizes, timing, association, source/classification labels, usage numbers and identities remain visible until purge; encryption does not hide these metadata.

## P2 schema

PostgreSQL types below are authoritative design types. SQLite uses TEXT, INTEGER and BLOB equivalents, checks Boolean values explicitly, enables foreign keys and uses WAL. `JCS` means validated canonical UTF-8 bytes in `bytea`, never plaintext opaque JSON. Unless stated otherwise, rows below have `tenant_id text NOT NULL, session_id text NOT NULL`; every primary key, unique key, foreign key and session index includes that tenant prefix. IDs are text. Sequence/time/count columns are nonnegative bigint, attempts are positive integer, digest columns are exactly 64 lowercase hex characters.

| Table | Columns and constraints |
| --- | --- |
| `sk_session` | PK `(tenant_id,session_id)`; `schema_version text CHECK = 'S1'`; `parent_session_id text NULL`, `parent_operation_id text NULL`; `head_seq`, `inbox_seq`, `lease_epoch` default 0; `lease_owner text NULL`, `lease_until_ms bigint NULL` paired; `phase text` in current `idle,active,parked,suspended,closed`; `active_turn_id text NULL`, `next_drive_ms bigint NULL`; `drive_retry_at_ms bigint DEFAULT 0`; `terminal_seq bigint DEFAULT 0 CHECK BETWEEN 0 AND head_seq`; `level integer >= 1`; `handler_version text NULL`; `required_capabilities JCS` sorted unique string array default `[]`; `deletion_state text` in `live,forgetting,forgotten,purging`; `created_ms bigint`. Parent FK `(tenant_id,parent_session_id)` references this table; parent IDs are both null or both non-null. |
| `sk_record` | PK `(tenant_id,session_id,seq)` with `seq>0`; `record_id text`, `schema_version text CHECK = 'S1'`, closed `kind`, nullable `turn_id,operation_id,attempt`; `body JCS` of the reference-form record; `digest text`, `chain_digest text`; `commit_id text`, `commit_digest text`, `commit_position integer>=0`, `writer_epoch bigint>=0`, `created_ms bigint`. Unique `(tenant_id,session_id,record_id)` and `(tenant_id,session_id,commit_id,commit_position)`. Session FK. |
| `sk_inbox` | PK `(tenant_id,session_id,input_id)`; `input_seq bigint>0`; closed `kind`; `body JCS` of reference-form input including actor; `digest text`; `admission_key_id text NULL`, `admission_signature bytea NULL`, `admission_signed_ms bigint NULL` (all null or all present); `available_ms,received_ms bigint`; `consumed_seq bigint NULL`. Unique `(tenant_id,session_id,input_seq)`; session FK and composite FK from consumed position to record. C slots use the same encryption as `input_applied.input`; the admission signature binds the full reference-form envelope, including clear fields. The proof is mandatory in signed deployments. |
| `sk_consumer` | PK `(tenant_id,session_id,consumer)`; `last_seq bigint DEFAULT 0`; `project_retry_at_ms bigint DEFAULT 0`; session FK. Consumer is a registered opaque name. |
| `sk_commit` | PK `(tenant_id,session_id,commit_id)`; `body JCS` containing only ordered record IDs/digests, consumed input IDs and creation consumer IDs; `digest text`; `record_sequences JCS` of ID/seq pairs; `head_seq bigint`; `created_ms bigint`; session FK. No requests, results or callback bodies. Preserve empty and fully overlapping receipts. |
| `sk_checkpoint` | PK `(tenant_id,session_id,seq)`; `chain_digest text`, `key_id text`, `signature bytea`, `signed_ms bigint`; composite FK to record at seq. Required at every record seq from creation in signed deployments. No content keys. |
| `sk_session_key` | PK `(tenant_id,session_id)` and unique `(tenant_id,key_id)`; `key_id text`, `provider_id text`, `wrapped_key bytea`, `state text` in `active,destroying`; `created_ms bigint`; session FK. Wrapped bytes are encrypted material, never a raw key. Provider root keys and revocation state are outside this database. |
| `sk_content` | PK `(tenant_id,session_id,content_id)`; `key_id text`, `mac text`, `size bigint`, `labels JCS`, `nonce bytea` (12 bytes), `created_ms bigint`; `state text` in `staged,live,gc_pending`; `pin_until_ms bigint`; `ciphertext bytea NULL`, `object_locator bytea NULL`, exactly one non-null. Ciphertext includes GCM tag. Locator is an opaque backend object key, never credentials or a signed URL. Unique `(tenant_id,session_id,key_id,mac,labels)` for session-local deduplication. Staged rows can predate session creation, so no FK requiring a committed session here. Final publication checks key/session ownership. |
| `sk_content_link` | Composite session FK; `content_id text`; `record_seq bigint NULL`, `input_id text NULL`, exactly one non-null; `slot text` a schema-owned content-slot path including typed array positions. Outer objects and all nested typed message references have flattened owner/slot entries. Unique owner/slot index separately for records and inputs. Composite FKs to the owning record/input and content object. These are reachability indexes, not another execution ledger. |
| `sk_tombstone` | PK `(tenant_id,session_id)`; `final_seq bigint`, `chain_digest text`, `deleted_ms bigint`; `actor JCS` with Principal partition below; `key_id text`, `signature bytea`; `state text` in `pending,complete`; `scope_root_session_id text`. No FK to the session it outlives. Signed material excludes mutable state. Never permits reuse of a deleted session ID. |
| `sk_settings` | One global row, PK `singleton integer CHECK = 1`; `storage_major text CHECK = 'S1'`; `write_level integer >= 1`. Runtime roles read it; only the operator role changes it. This is the sole non-session, non-tenant table. |

Indexes: execution scan `(level, next_drive_ms, tenant_id, session_id)` for live rows with due time; parent lookup `(tenant_id,parent_session_id)`; pending inbox `(tenant_id,session_id,available_ms,input_seq)` where unconsumed; operation `(tenant_id,session_id,operation_id,attempt,seq)`; turn `(tenant_id,session_id,turn_id,seq)`; consumer lag/retry; content GC `(state,pin_until_ms,created_ms)`; deletion jobs via deletion state and pending tombstones. All scheduling predicates also check the lease, retry gate and worker requirements; an index is not authority.

Session row head/phase/schedule/requirements are transactionally maintained indexes of the log. Rebuild must reproduce them from retained S1 records; deletion state is administrative state, not a reducer result. Inbox insertion, applying it, content-link publication, level changes, child admission, reservations and commit receipts retain their existing atomic boundaries.

## Field disposition: every current record

The envelope fields `schema_version, record_id, kind, session_id, turn_id, operation_id, attempt` are S. SQL envelope fields `seq, digest, chain_digest, commit_id, commit_digest, commit_position, writer_epoch, created_ms` are S. Every field listed below is a current payload field, including optional ones; nested partitions follow. `input_applied.input` uses the full inbox partition, not one unencrypted copy.

| Record | S payload fields | C payload fields / nested partition |
| --- | --- | --- |
| `session_created` | `parent_session_id, parent_operation_id` | `principal` becomes S/C Principal; `workspace` C; `attributes` C in full. |
| `turn_started` | `input_ids, definition_digest, handler_version, binding, generation` | `definition, budget` C. Frozen typed requirements and environment bindings are new S/C fields below. |
| `input_applied` | `input_digest, disposition, target_operation_id, target_wait_id, position` | `input` S/C inbox; `reason, model_binding` C. |
| `op_planned` | `op_kind, purpose, request_digest, context_version, dependencies, idempotency_key, not_before_ms, provider_binding, consumed_unknowns` | `request, tool, budget_admission` C; message extraction applies to model requests. |
| `op_prepared` | `request_digest, op_kind, provider_binding, idempotency_key` | `request, tool, hook_result, shared_state` C. |
| `turn_parked` | `interaction_id, source_operation_id, source_attempt` | `question` C, even when short. |
| `op_started` | `dispatch_id, authorization_version, epoch, mode, endpoint_id` | None. Endpoint identity is an opaque route ID, never a URL. |
| `op_parked` | `phase, poll_at_ms, deadline_ms, yield_on_steer, yieldable_children` | `handle` S/C below; `interaction_result` C; `delegation` S/C below. |
| `op_completed` | `result_digest, execution_started, context, request_digest, provider_binding, yielded_by` | `result, usage, shared_state, evidence` C. The typed numeric usage projection is defined below. |
| `op_unknown` | `retry, retry_at_ms, duplicate_cost_risk, measurement_missing` | `reason, dispatch_evidence, observation` C. Unknown does not imply zero usage. |
| `context_compacted` | `source_digest, prefix_ids, tail_ids, mode, summary_operation_id, micro_usage` | `replacement` C as message references; `evidence_manifest` C, including artifact paths and plaintext hashes. |
| `boundary_recorded` | `boundary_id, stage, source_operation_id, source_digest` | `data` S/C by stage below. |
| `usage_observed` | `meter_id, observation, mode` | `usage, source` C; arbitrary provider source text is not a kernel enum. |
| `turn_ended` | `status, adopted_results, unconfirmed_operations` | `reason, result, budget` C. |

`ResultId.operation_id, ResultId.attempt` and all three `micro_usage` fields (`archived_count, reclaimed_tokens, artifact_failure_count`) are S. Generic `usage` and `budget` objects are C even if a particular fixture contains only numbers: their current schema permits arbitrary strings and provider data. Numeric accounting remains available through the typed projection below, with the encrypted upstream usage preserved byte for byte for authentication and duplicate checks.

### Every current inbox field

Envelope fields `schema_version, input_id, kind, target_turn_id, generation, available_ms` are S. SQL `input_seq, digest, received_ms, consumed_seq` are S. The owning session/tenant and required `actor` come from trusted host/provider authentication. Signed admission then binds those values and the complete stored input before publication, including provider evidence and controls. The three admission signature columns are S. Mutable `consumed_seq` is outside the admission proof and must agree with signed consumption records.

| Inbox kind | S payload fields | C payload fields / partition |
| --- | --- | --- |
| `user` | None | `content` C, including typed reply `interaction_id,text,operation_id` if present inside that opaque content. |
| `steer` | None | `content` C. |
| `follow_up` | None | `content` C. |
| `model_selection` | None | `selection` C or null; all model settings, route and binding content stay encrypted. |
| `provider_result` | `operation_id, attempt, request_digest, provider_binding` | `result, usage, evidence` C. Authentication covers all three hydrated values plus actor/target. |
| `approval_answer` | `operation_id, attempt, request_id, request_digest, decision` | `scope, reason, metadata` C. Scope may contain argument predicates, not just tool names. |
| `child_result` | `session_id, turn_id, operation_id, attempt, status, terminal_seq, terminal_digest` | `result` C; parent owns a re-encrypted copy under its own key. |
| `control` | `action` | None. |
| `provider_evidence` | `operation_id, attempt, request_digest, yield_on_steer` | `handle` S/C below. |

`input_applied.input` reuses these exact references and actor values. Its digest is the digest of the stored reference-form input. The new S `input_proof` copies the original input sequence, received time and signature proof; consumption never generates a new admission signature, plaintext copy or content identity.

### Every current nested handle and delegation field

| Type | S fields | C fields |
| --- | --- | --- |
| Provider handle | `kind, provider, operation_id, attempt, request_digest` | `job_id, evidence, query_ref, cancel_ref`. Existing handles may contain URLs or capabilities; opaque host strings are not safe by default. Remote execution adds a validated stable execution ID separately. |
| Approval handle | `kind, request_id, request_digest` | `scope`. |
| User handle | `kind, interaction_id` | `question`. |
| Child handle | `kind, session_id, turn_id, generation, background, delivery_target, siblings` | None; delivery target children `session_id,turn_id,generation,operation_id,attempt` and sibling children `session_id,turn_id,generation` are all S. |
| Delegation | `mode, handoff_count, max_handoffs` | `agent_name, metadata`. |

### Every current boundary field

| Stage | S data fields | C data fields / partition |
| --- | --- | --- |
| `before_memory` | None | `messages` as message references; `shared_state` C. |
| `after_cycle` | `action, disallow_tools` | `steering_messages, shared_state, error` C; `stop.code` S only as a closed kernel code, `stop.message` C. Host-defined stop codes remain C with an S generic code. |
| `memory_started` | None | `event` C in full. |
| `memory_completed` | None | `event` C in full. |
| `session_memory_saved` | `state.last_extracted_message_index, state.tokens_at_last_extraction, state.initialized`; each entry's `category, source_cycle, importance` | Each `state.entries[].content` C. Array order remains significant. |
| `output_checked` | `status` | `reason, value, partial_output` C. |
| `budget` | `tool_names` | `usage, exhaustion` C. |

### Typed content below opaque boundaries

The following lists are exhaustive for currently codec-validated content subtrees. Each listed descendant is C because its owning field is C; there is no clear duplicate. They remain typed and validated after hydration.

| Current subtree | Encrypted fields |
| --- | --- |
| `attributes.seed` | `unseen_input_protection, response_rejection, messages, shared_state`. Preserve independent default-on-at-first-creation rules and replay/inheritance; do not reinterpret an absent choice during S1 reads. |
| `attributes.app_server` | `agent_key, cwd, metadata`. |
| `attributes.child_admission` | `mode, selector, definition, definition_digest, budget, handler_version, sub_config, exclude_files_pattern, handoff_count, max_handoffs, handoff_metadata`. |
| `child_admission.sub_config` | `model, description, backend, system_prompt, max_cycles, session_memory_enabled, exclude_tools, metadata, denied_side_effects, denied_capability_tags, deny_terminal_tools, denied_cost_dimensions`. |
| Retained `attributes.child_handle` | Entire child-handle subtree above is C here; it stays content under opaque attributes. The indexed admission/delivery skeleton uses separately validated child IDs; do not promote arbitrary attributes. |
| `definition` / task `metadata.vv_session` | `host_binding_names, max_handoffs, handoff_targets, input_messages, memory_initial_state, input_blocked`, plus all AgentTask/PromptBundle bodies and host metadata. |
| Request `metadata.vv_session` | `endpoint_order, endpoint_id, shared_state, cycle_index, model_binding`. |
| `ModelBinding` | `backend, model, endpoints, context_length, max_output_tokens`. |
| `ModelSelection` | `model, model_settings, route`. |
| `ModelSettings` | `temperature, top_p, max_tokens, tool_choice, parallel_tool_calls, reasoning, response_format, timeout_seconds, retry, extra_headers, extra_body, extra_args`; nested `tool_choice.type,function.name`, `response_format.type,json_schema`, `retry.max_attempts,backoff_seconds` are also C. Credentials must never be accepted in these persisted values, even encrypted. |

`Message`, `ToolExecutionResult`, provider raw bodies, archive manifests, file/resource references and arbitrary host JSON retain all of their current inner fields as C. Their already encrypted owning boundary covers any further descendants. New S projections are allowlisted typed facts, never a second dump of opaque content.

## Reserved fields and activation

Counter/tokenizer binding is also reserved at the sole S1 boundary. In the inspected vv-llm 0.10.1, `_get_qwen_tokenizer` caches every Qwen model under `"qwen"`; qwen-tokenizer 0.3.0 selects different vocabulary assets for Qwen 3.5/3.6. The first initialized model therefore selects the shared object and can affect later counts. P1 extraction preserves that selected-object behavior and does not establish cross-restart counting determinism. P2 must freeze an explicit binding in the runtime definition covering model identity, counting method/implementation, tokenizer asset identity and input normalization, with matching compatible-worker requirements. Acceptance must exercise differing initial-model orders and restarts: workers reproduce the frozen binding/counts or remain incompatible before replay/dispatch. Exact canonical fields, identities and normalization/default allocation remain P2 decisions. Reuse the authoritative settings/defaults; add no historical decoder, parallel runtime framework or vv-llm change in P1.

P2 installs these typed slots once. Identity and integrity fields marked required are active in P2. Later optional slots have explicit absent/null/default behavior; activating them or adding kinds/enums requires an assigned feature level and actual producer/consumer behavior. A reserved nullable slot is not a delivered security feature.

| Location | Shape / P2 default | Activation |
| --- | --- | --- |
| Creation `principal` and inbox `actor` | Required `{tenant_id,subject_id,subject_kind,attributes}`. First three S; `attributes` C. Immutable creation principal; actor tenant equals owning tenant. | Required in P2; single-tenant hosts use `tenant_id="default"`, never an inferred tenant. ToolContext gets read-only principal; model adapters receive it outside prompts. |
| Inbox admission proof / `input_applied.input_proof` | SQL proof columns above; the retained record proof is null or closed `{input_seq,received_ms,key_id,signature,signed_ms}` S. JSON signature bytes use canonical base64; SQL uses raw bytea. | Required in signed P2 deployments; null only in explicitly unsigned local mode. Copy the original proof on consumption/export and verify it without content keys after forget. |
| `sk_checkpoint` | One existing checkpoint shape for each record seq, including `session_created`. | Signed P2 uses N=1 from creation. Missing/invalid required signatures reject recovery and extension; no retro-signing. |
| `sk_session` / `turn_started.requirements` | `{level,handler_version,capabilities}` with opaque handler ID and sorted unique capability IDs; session level starts 1; pre-turn handler may be null only under the bootstrap rule in the execution design. | P2 writes/validates levels; W4 performs worker support filtering. P2 needs a correct admission filter already to meet W2a rollout acceptance. |
| `ContentRef.labels` | `{source:[], classification:"internal"}` while W7 inactive. | W7 derives source sets and maximum classification. Unlabelled historical content is conservatively untrusted at activation; activation must not assert trusted provenance for P2 bodies. |
| `turn_started.policy_digest` | S reference-form digest or null; full policy definition is C. | W6a fixed baseline then W6b policy; null means no W6 claim. This is the ledger domain, not an operational plaintext policy hash. |
| `turn_started.environment` | Null or closed `{descriptor, capabilities, capabilities_digest}`. `descriptor` C; capabilities partition below. | W3. |
| Environment capabilities | S `environment_id,generation,isolation,network,enforces_network_policy,enforces_resource_limits,persistent,network_policy_ref_digest,isolation_policy_ref_digest`; C `network_policy,isolation_policy,evidence`. Each policy object contains its canonical document and operational digest. | W3 freezes and rechecks, including at command start. Ledger policy/evidence digests cover reference forms; operational hashes stay encrypted. Exact new field allocation is pending canonical acceptance. |
| Operation plan / preparation / completion `policy_decision` | Null or `{policy_digest,rule_id,effect}` S; effect is `allow,deny,require_approval`; any explanatory prose/attributes C. | W6a/W6b; recovery uses retained decision. Approved request changes require a new binding. |
| Approval answer | Optional S `policy_digest,rule_id,expires_ms`; actor above identifies approver; scope predicates C. | W6a; absent expiry does not remove the existing parked approval deadline. Later activated rules may require these fields on new answers. |
| Command plan | Optional S `environment_id,generation,execution_id,network_policy_ref_digest,isolation_policy_ref_digest,command_digest`; command/descriptor/evidence and the expected operational policy tuple C. | W3; per-attempt request digest remains distinct from stable command digest. Hydrate the operational tuple for remote start; never send the ledger reference digests as operational hashes. |
| Record usage projection | Optional `usage_counts` with the closed numeric projection below. Arbitrary provider names/text and host unit/currency strings stay C. | P2 must define the metrics used by existing billing/budgets and prove equality to authenticated encrypted usage; source bytes remain C. No zero filling. |
| Creation lineage | Optional `migrated_from={session_id,chain_digest}` S; optional fork lineage uses IDs/cut digests only. | Future major migration; never grants access to source content. |

`usage_counts` has the following closed shape; optional absence means no skeleton accounting projection, not zero usage:

```text
usage_counts = {tokens: TokenCounts | null, budget: BudgetCounts | null}
TokenCounts = {
  input_tokens, output_tokens, total_tokens, reasoning_tokens: nonnegative integer | null,
  usage_source: provider_reported | estimated | accounting_missing,
  cache_usage: {
    status: provider_reported | accounting_missing | unsupported,
    read_input_tokens, write_input_tokens, uncached_input_tokens: nonnegative integer | null
  }
}
BudgetCounts = {
  cycles, tool_calls, elapsed_ms: nonnegative integer,
  total_tokens, uncached_input_tokens: nonnegative integer | null,
  tool_calls_by_name: {registered_tool_name: nonnegative integer},
  host_cost: {amount_microunits: nonnegative integer, unit: ContentRef,
              currency: ContentRef | null} | null,
  unavailable_dimensions: [{dimension, reason}]
}
```

`dimension` and `reason` use exactly the current BudgetDimension and BudgetUnavailableReason enums in `src/vv_agent/budget.py`. Arbitrary cache `source`, `provider_usage`, model/backend display strings and unavailable-dimension unit/currency details remain C in upstream usage. Only numeric/enumerated leaves are S; the host unit/currency slots are C. Produce the projection with existing usage normalization; do not reinterpret or alter authenticated upstream usage. Apply current TaskTokenUsage aggregation rules, including unavailable cache totals and all audit attempts. P2 fixtures cover TokenUsage, CacheUsage, TaskTokenUsage and BudgetUsageSnapshot; the encrypted usage remains authoritative for duplicate authentication.

## Tenant authorization

`store.for_tenant(tenant_id)` scopes every method: create, push, append, reads, state/context projection, all inbox paths, consumer batches/ack, acquire/renew/release/status, scheduling, exports, content, checkpoints and deletion. A caller-supplied reference is resolved against authenticated tenant and owning session/key, not just `content_id`. Authenticated same-tenant subject/resource authorization stays at the host boundary; RLS alone does not authorize two users within one tenant. Children inherit tenant and immutable identity rules; IDs never imply organization membership.

Tenant isolation does not authorize one user to access another user's session inside the same tenant. Organization membership, resource grants and execution-time authorization remain backend responsibilities. User-facing read, write, control, export, deletion and replay/subscription paths must authorize the authenticated Principal for that session and action before accessing the tenant-scoped store. Raw SessionStore access is a trusted service interface, not an end-user authorization boundary.

App Server integrations reuse `AppServerHost.authorize` in `src/vv_agent/app_server/host.py`, which the request dispatcher already invokes, and the existing resource-authorization hook for retained artifacts. Resolve Principal from trusted connection/request authentication, never from model arguments or caller-supplied identity fields. The default host explicitly trusts local single-user connections; an enterprise deployment must supply its backend grant checks. W5 must preserve these checks across direct reads, listing, reconnect replay and new subscriptions instead of treating tenant membership alone as permission. This adds no parallel account or grant system to vv-agent.

All session tables, including content links, keys, checkpoints and tombstones, use PostgreSQL ENABLE and FORCE RLS. `SET LOCAL vv.tenant_id` is request context, not authentication: any ordinary database client can set a custom GUC to another tenant. A policy comparing only that GUC to tenant_id does not meet the plan's database-credential isolation requirement.

For that requirement, install a distinct authenticated PostgreSQL LOGIN role per tenant, with policies pinned to both that immutable login and a tenant constant. For tenant A, the generated policy on each session table is equivalent to:

```sql
CREATE POLICY tenant_a_scope ON sk_record TO tenant_a
  USING (session_user = 'tenant_a'
         AND tenant_id = 'A'
         AND current_setting('vv.tenant_id', true) = 'A')
  WITH CHECK (session_user = 'tenant_a'
              AND tenant_id = 'A'
              AND current_setting('vv.tenant_id', true) = 'A');
```

The installation script generates the same binding for every table and each tenant using correctly quoted identifiers/literals. The tenant login is NOSUPERUSER, NOBYPASSRLS, NOCREATEROLE, not a table/schema owner, and has no role membership permitting tenant B, worker or owner impersonation. It cannot change session authorization to another login, alter policies, grant roles, or mutate the role-to-tenant installation. `session_user` rather than `current_user` prevents a SET ROLE change from replacing the authenticated binding. Grants permit only required DML, never TRUNCATE/DDL. Parent/content/consumer/commit constraints include tenant_id. No SECURITY DEFINER function is installed.

Pool physical connections by authenticated tenant login. Reusing A's connection with `SET LOCAL vv.tenant_id='B'` sees no B rows and cannot write B; serving B requires a B-authenticated connection. Always open an explicit transaction and set context with SET LOCAL; missing context fails closed. A pool must not use a superuser/owner connection and SET SESSION AUTHORIZATION to emulate tenant logins. FORCE RLS is not a defense against a superuser or a role allowed to alter its own table, so runtime startup rejects such credentials.

A trusted gateway using a shared cross-tenant application login can also set transaction-local tenant context, but that authenticates the gateway, not the tenant at the database. Its context policies prevent accidental query leakage; a holder of those shared credentials can impersonate any tenant the gateway serves. This is a separate trust mode and cannot claim that tenant A credentials are database-isolated from B. Deployments claiming the plan's A-credential property must use the bound tenant roles above. Cross-tenant workers use a distinct explicitly granted worker policy/login and credentials unavailable to tenant paths; they remain outside that threat boundary. Owner/migration credentials run only schema administration. Global settings have read/operator grants and no tenant payload. SQLite enforces scope in the store, without claiming database RLS.

Keys, HMAC lookup, object reads/listing and signed references must be tenant scoped too. Neither object naming nor encryption alone replaces authorization. Disclosure through worker credential compromise remains outside the approved shared-worker threat model.

## Digests, chains and signatures

Define `D(x)=SHA256(JCS(x))` in lowercase hex, with every C value replaced by its committed reference representation. Apply this to record/inbox, definition, request, result, context source, child terminal, commit and derived IDs. Structural digests remain keyless; content bytes are protected by the session HMAC and AES-GCM. Evidence/reference digests cannot authenticate an external provider by themselves.

Policy hashes have two distinct domains. For remote equality, `network_policy_digest` and `isolation_policy_digest` are ordinary `SHA256(JCS(policy_document))` over the complete effective operational policies. The trusted provider supplies policy objects `{document, operational_digest}` and verifies that equality; the documents include allowlisted host:port pairs and effective limits. Persist those objects and any expected operational tuple only as C. Private operational hashes must not appear in clear SQL columns, skeleton records, derived IDs, ordinary logs or skeleton exports.

For the ledger, `network_policy_ref_digest=D(ContentRef(network_policy_object))` and `isolation_policy_ref_digest=D(ContentRef(isolation_policy_object))`. Generic `policy_digest` fields likewise cover the committed reference form of their policy definitions. These hashes bind session-owned encrypted references, not guessable plaintext policies. The worker hydrates the operational hashes before attach/start comparison. The remote agent computes ordinary canonical policy hashes and needs no session encryption or HMAC key. Command fingerprints remain the separate session-keyed HMAC domain. Canonical fixtures must allocate and distinguish these fields before implementation.

A repeated source submission must resolve to the same session-local references before byte comparison. Deduplication uses `(session key, MAC, labels)` and returns the original immutable object/reference, even with different encryption nonces on a competing staging write. Canonical references and message order are stable across retries. If the key is unavailable, defer before duplicate comparison; do not invent a new receipt. After forgetting, normal admission/read is refused, but explicit skeleton export/verification remains available.

For chain framing, concatenate 32-byte binary digests, unsigned 64-bit big-endian seq, and UTF-8 identifiers with unsigned 32-bit byte-length prefixes:

```text
C0 = SHA256(UTF8("vv-agent/S1") || LP(tenant_id) || LP(session_id))
Cn = SHA256(Cn-1 || U64BE(n) || HEX_DECODE(record_digest_n))
```

No decimal-string concatenation or ambiguous identifier delimiters. Chain fields live in the SQL envelope, excluded from record identity. Append computes the chain under the same session lock, lease fence and transaction as log writes. Identical overlapping records do not extend it. Verify recomputes canonical body digest as well as chain; it checks contiguous positions and SQL/log identity agreement.

Checkpoint signing message is JCS `{domain:"vv-agent/S1/checkpoint",tenant_id,session_id,seq,chain_digest,key_id,signed_ms}`. Signed deployments use N=1: every record, including `session_created`, has a checkpoint in its append transaction. Sparse 256/turn-ended signing is not an adequate signing default and cannot satisfy the approved historical-tamper goal.

The minimal future signer sketch is `bound = signer.bind_key()`, exposing immutable `bound.key_id`, then `signature = bound.sign(message)` returning bytes. Construct the message with that bound key ID before signing. The binding pins the actual signing key across rotation; signing must fail rather than silently switch keys. Verification uses the stored key ID and exact framed message. This is a proposed canonical interface detail, not an existing exported API or a call to the plan's `sign(message) -> (key_id,signature)` with an unknown key ID embedded in its argument.

Signatures and checkpoint/input-proof insertion participate in the original create, append or input-admission transaction. Signer failure rolls back that whole transaction. Key binding and signing use the bounded SDK persistence budget below, followed by final lease/CAS checks where applicable. Signed mode is required by trusted deployment configuration, never inferred from mutable database flags or missing proofs; there is no unsigned fallback. Explicit hash-only local mode reports consistency only and cannot claim database-tamper detection, historical authenticity or authenticated input/approver attribution. Its old history cannot be retro-signed into those assurances. No second signing queue or cross-system transaction is introduced.

Before recovery, extension or use of retained history, verify canonical body digests, chain/identity consistency and every required record signature in the history being trusted, including the predecessor head. Missing or invalid proofs stop the operation. A later signer may sign new trusted records only; it must not repair an existing unsigned tail using database evidence alone. Negative case: checkpoint 256 survives, a database writer changes an actor in records 257–260 and recomputes their unsigned hashes, then an honest worker tries to append/sign. The signed deployment must reject before creating a new checkpoint; later honest signing cannot establish the old bytes' provenance.

Input admission reuses the same bound LedgerSigner, with a distinct message domain:

```text
JCS({domain:"vv-agent/S1/input",tenant_id,session_id,input_id,input_seq,received_ms,body_digest,key_id,signed_ms})
```

`body_digest=D(stored_input_body)` covers the complete immutable reference-form input: schema/kind/identity, target turn and generation, available time, actor, control action, approval decision/scope, request and all content references. The SQL kind/identity/available time must agree with that body. Allocate input_seq and received_ms in the original admission transaction, bind the signing key, then sign those trusted submission values and publish the input plus proof atomically. These are the original received values, not a later copy reconstructed from untrusted SQL. Duplicate submissions verify and return the original sequence/time/proof; they do not acquire a new signature or identity. Existing provider-result authentication and host authorization still apply.

Before decrypting/hydrating a stored input or applying its control/approval, recompute its body digest and verify the complete envelope and signature with the retained public key. Invalid/missing proof is an integrity failure; it cannot be converted to an unsigned input or repaired at consumption. After verification, `input_applied` records the unchanged input and `input_proof`; its own per-record signature binds both. Retain the original body and proof in inbox rows, consumption records and exports. Forget removes content keys/bytes but retains these reference-form facts and public verification keys: signature verification remains possible without decrypting erased scope/request content. Admission proof authenticates retained inputs; it does not assert their later consumption or that no pending input was deleted.

Retain verification public keys across signer rotation and session forget for every retained record, input proof, tombstone and export. Valid required per-record signatures authenticate the retained history from creation, given trusted verification keys. Keyless chain validation alone detects local inconsistency, but a database writer can recompute an unsigned chain. Neither proves the retained checkpoint is the latest: deleting an entire trailing signed checkpoint and its suffix can leave a valid earlier prefix. Detecting that truncation requires the existing independently retained expected head/checkpoint (explicit export or trusted host anchor), covering the history whose completeness is claimed. A head read from the same database or an older anchor does not prove latestness. Verification reports chain consistency, authenticated-through seq, input-proof validity and completeness relative to that expected head. Without it, do not claim rollback detection or ledger completeness. Check a supplied expected head before recovery/extension as well as offline verification; reject conflicting or truncated history. This retains the existing external-head boundary and adds no witness service or 2PC.

## Keys and content publication

One random 256-bit data key belongs to one `(tenant,session,key_id)`. AES-256-GCM uses a fresh 96-bit nonce per encryption under that key. Authenticate JCS `{domain:"vv-agent/S1/content",tenant_id,session_id,content_id,key_id,mac,size,labels}` as AAD. GCM encrypts with the session key; HMAC uses the same session content key as specified by the plan, with a separate algorithm purpose. Never share a data key or deduplicate plaintext across sessions/tenants.

Creation sequence: allocate the provider's session key, persist encrypted initial content, then commit session, wrapped-key row, creation record, consumers and reachability links in one SQL transaction. Stage before the transaction when the inputs are already known. When a child's IDs/content are discovered by the existing admission callback, SDK persistence may run inside that same transaction under the budget below. Do not independently commit business rows or require another child preparation API. Failed creation leaves only unreachable encrypted objects/provider key allocations; an orphan job removes them. An allocation is not a usable session until the whole admission transaction commits.

Content publication sequence:

1. Allocate or find the session-local immutable content object; persist ciphertext and its metadata before a record/inbox can reference it. Persist a bounded staging pin using database time. Upload S3 content before registering its usable locator; never persist a signed URL.
2. In the admission transaction, lock the settings row for reading, session rows in stable `(tenant,session)` order, then content rows in ID order. Verify live session/key, level ceiling, usable pins and object identity; register all flattened links and transition referenced rows to live with log/inbox publication. A pin that expired or an object already claimed by GC must be restaged before admission. Narrowly scoped SDK key allocation/unwrap, content persistence and record/input signing may run here when not already staged. Input sequence/time allocation and its proof remain inside this original admission transaction.
3. After all persistence/signing work, revalidate owner/epoch, database-time lease expiry, head/inbox CAS, deletion fence and current write-level/requirements before admission can commit. Expiry, cancellation that invalidates admission, integrity error, unavailable service or validation failure rolls back the entire SQL unit, including host business rows and every child.
4. Commit; only then wake execution/projection. A crash before commit leaves only scoped orphan allocations/ciphertext. A lost commit reply is handled by the existing commit/input receipt; it is not a fresh operation.

Use a cumulative admission deadline within the last confirmed lease's remaining safety budget, reserving time for final validation and outer COMMIT. Bound each SDK request and retry by that remaining deadline; SQL statement timeouts alone do not bound KMS/object calls. A heartbeat cannot renew a session row held by this write transaction. Insufficient budget means rollback/defer, never optimistic admission. Backend allocations/uploads, including late completion after a timeout, remain scoped to orphan cleanup unless the referencing SQL admission committed; GC pins and deletion fences still apply. During in-transaction persistence, backend-visible ownership/pins protect active allocations through the admission deadline; orphan cleanup cannot rely on a SQL pin that is still uncommitted. External persistence does not join the SQL transaction.

This exception is for SDK-owned persistence, not arbitrary host callback I/O, provider/tool invocation, child execution or irreversible business effects. Host callbacks remain short database checks/writes on the driving connection. At implementation, explicitly qualify `docs/host-integration.md`'s existing file/network-I/O prohibition for this bounded SDK persistence only. Keep the same callback signatures per the specific W4.6 rule; no mandatory discovery/rollback/prepare/rerun protocol is introduced.

`KeyUnavailable` at put/get/unwrap defers drive or projection with a bounded retry gate, like RuntimeNotReady, without consuming the input or failing the turn. Renewal uses a separate connection. Crypto code emits only safe codes and opaque IDs; exceptions and chained exception text must not print plaintext, key bytes or provider credentials. Caches live only within a drive and are dropped on exit, forget or lease loss. Production debug dumps/core dumps and plaintext spooling must be disabled for this path.

## Session-data inventory and deletion

P2 implements the plan's single code-owned inventory used by forget, purge and a schema coverage gate. It is a fixed list of owned tables/artifact stores and deletion order, not a plugin or generic lifecycle framework. Missing ownership for a new table/file/object fails the gate. The concrete content-link index is needed to prevent GC deleting a newly published reference; staging pins protect content before publication.

| Inventory entry | Forget | Purge / owner boundary |
| --- | --- | --- |
| `sk_session`, `sk_record`, `sk_inbox`, `sk_consumer`, `sk_commit`, `sk_checkpoint` | Retain reference-form skeleton, per-record signatures and original inbox/embedded input proofs; block execution/content reads. | Delete all owned rows/proofs, child-first where FK order requires; retained exports remain separate copies. |
| LedgerSigner verification public keys | Retain independently of destroyed session content keys. | Keep while any retained input/record/tombstone/export needs verification; purge does not erase these public keys. |
| `sk_content_link` | Remove links after key destruction and fencing, before deleting content metadata. Ledger references remain immutable. | Delete remaining links before owner/content rows. |
| `sk_session_key`, provider session-key registry / Aliyun key capability | Destroy capability, invalidate worker caches, remove wrapped row. | Same; retries must finish even after SQL row deletion. Never infer successful destruction from a missing SQL row. |
| `sk_content` and S3 ciphertext, incomplete uploads, staging objects | Delete ciphertext and metadata after key destruction. | Same, including unreferenced objects scoped to these sessions. |
| `sk_tombstone` | No persistent purge tombstone required for forget; `sk_session.deletion_state` journals progress. | Keep only signed deletion facts and pending/complete state; completion leaves the signed tombstone. |
| `sk_settings` | No session content. | Retain global row. |
| In-memory fold/message/hydration/key caches | Evict on deletion fence and before reporting completion. | Same; acknowledgments/quiescence prevent a serving process retaining usable keys. |
| Runtime-owned archived tool bodies, compaction artifacts, Session Memory projections and retained App Server resources | Canonical bodies must reside in encrypted content storage. Disposable projections must be inventoried and removed. | Same; do not leave plaintext `.memory` files or archive mirrors under worker temp paths. |
| Workspace files, remote command output/spool, environment snapshots | If runtime owns a copy, encrypt/register/delete it as above. Host workspace and environment retention require host cleanup receipts. | Host-owned outputs are explicitly outside kernel deletion; report them, never claim all-system deletion. |
| JSONL sinks, logs, traces, temporary directories, diagnostics | Default exports contain skeleton only. Body export requires explicit authorization and is a separately retained copy. Never emit content through ordinary diagnostics. | Previously delivered audit exports, provider retention, backups and host business tables remain separate retention domains. |

Backups containing ciphertext remain safe only if provider revocation prevents restored wrapped keys from working. Backups of local key registry material can defeat erase; either exclude revocable session secrets from restore or apply durable revocations before serving restored databases. Already exported plaintext cannot be recalled. This limitation must appear in host documentation and deletion evidence.

Forgetting is destructive even if an operator requests it on an active session. Fence new execution, input admission, projection and content puts first; stop/abandon activity explicitly without claiming an unknown remote effect is stopped. Automatic retention skips active, parked, unresolved, referenced or undelivered sessions and any unacknowledged consumer. It never silently abandons them. The reference pin includes pending child delivery and fork/copy preparation; idle age alone is insufficient.

Deletion is an idempotent state machine:

1. Lock scope, check authorization, fence admission/reads and increment lease epoch. For purge, enumerate the root and all descendants under those locks, insert signed pending tombstones with each final head and deleting actor, and mark `purging` in the same transaction. Freeze the descendant set by fencing child creation. For forget, mark `forgetting`. Only the deletion worker can read administrative progress; normal APIs reject all content-bearing access. Public skeleton verification is an explicit separate API.
2. Quiesce in-flight trusted workers and drop content/key caches before acknowledging erase. Refuse a late content publication by the deletion fence. Remote stop uncertainty stays recorded; the host handles external environment cleanup separately.
3. Destroy each session-key capability, durably record provider acknowledgment outside the recoverable wrapped-key material, and retry on KeyUnavailable. A crash after destruction but before acknowledgment must be safely repeatable. Do not report completion while revocation is unconfirmed.
4. Remove the fenced scope's content links, then delete ciphertext, pending uploads, runtime-owned projections and content metadata. Repeat list/delete until empty for the fenced scope. For forget, set `forgotten` and remove the wrapped key. Ledger references deliberately remain dangling and still verify.
5. For purge, delete links, consumers, inbox, checkpoints, commit metadata, records, key rows and session rows in one final SQL transaction, descendants before parents; mark tombstones complete. Do not delete tombstones or signing public keys. The key/ciphertext destruction steps cannot be rolled back by SQL and need no false cross-system transaction.

The tombstone signs JCS `{domain:"vv-agent/S1/purge",tenant_id,session_id,final_seq,chain_digest,deleted_ms,actor}`. Actor identity uses Principal's S fields; attributes, if retained, remain a reference to already erased content. Tombstone signatures use the same bound-key signing procedure, with `bound.key_id` stored beside the signature, and a signing key separate from erased content keys. Purge requires a configured signer and refuses before fencing if none is available. Default cascade includes descendants, not unrelated sessions; surviving parent histories may still mention child IDs under their own retention. Signed deletion metadata itself is the approved exception to full metadata removal.

GC uses the inventory and flattened content links, including message objects referenced inside encrypted containers. A live outer object is not a substitute for those links. It locks candidate content metadata, rejects linked/live or unexpired pinned objects, marks an eligible unlinked object `gc_pending`, then deletes object bytes and metadata idempotently. Publication and GC share that lock; publication refuses `gc_pending`. A grace period alone is not a concurrency fence. Lost-upload objects not yet registered in SQL are removed by backend listing plus age/pin ownership checks. Unreferenced provider keys are collected only after proving there is no committed key row or live allocation pin. GC never decides an execution outcome.

Forks/clones allocate a new key, re-encrypt all copied content with new references/MACs and copy provenance labels. Cross-session ContentRefs are rejected even inside one tenant. The target commits only after all required copies exist; failed preparation leaves target-owned staged garbage. Child final results copied into a parent likewise become parent-owned content. Independent forget tests must cover both directions.

## Feature levels, writing and claims

Each optional field, enum value or record kind belongs to one contract feature level with a canonical absent/default behavior. Levels are monotonically assigned integers inside S1. Unknown fields/kinds/enums still reject. There is no historical decoder or catch-all extension for kernel fields.

`sk_session.level` is the highest level present or required anywhere the worker must read for that session, including pending inbox inputs and relevant child-delivery dependencies. It never decreases. Admission computes the required level before committing new bytes and raises it in the same transaction. A high-level inbox input cannot sit behind a low-level session row. A transaction creating a child or dependency that a parent must inspect raises the parent's required read level atomically as needed. Validation rejects understated levels; rebuild verifies cached requirements against the log and inbox.

`write_level` is a database-wide ceiling, not the worker's maximum and not a reader selector. Every producer (host create/push, worker append, child delivery, forks, administrative writes) checks the global setting in its transaction. New workers emit the old shape while the ceiling remains low. Requests requiring a disabled feature are rejected before publication with an explicit unsupported-feature response; they are not silently stripped or parked as malformed content. Raise the setting only after the operator confirms all required producers/readers are upgraded. No registry is introduced. Do not lower it below any retained session's level or claim rollback safety after higher-level bytes exist.

The execution design supplies the exact supports/requirements predicate. P2 must apply at least major/level filtering before decoding, including targeted acquire and projection consumers; W4 adds full handler/capability filtering. Frozen requirements are logged in turn admission and cached on the row. A pre-turn session has immutable bootstrap requirements established by trusted host creation; the first turn replaces the execution requirements atomically. Selecting a later model or admitting a new tool/environment requirement uses the same transactional update, never a discovery failure after dispatch.

The lock order for writes is global settings, sessions sorted by tenant/session, then content metadata. A raising settings transaction conflicts with active writers, so no write can straddle the ceiling change without checking it. Lease-only operations do not need the settings lock. Incompatible workers skip before body decode, lease acquisition, callbacks or provider I/O. None available means pending work, not a failed turn.

## One-time transition and exports

P1 simulation and cross-review evidence are prerequisites to P2. P2 updates canonical schemas/fixtures first, then Python producers and the repository/versioning rules in the same release. Rust remains frozen at contract 23 / 0.21.x. S1 is planned for Python 0.40.0 with a contract major; this design does not allocate that contract release number.

For .ai only, follow the current [upgrade procedure](../upgrading.md#upgrade-procedure): stop new sessions/turns/inputs on the old pool; drain with pinned 0.35.0; persist product history while that runtime can still read it; inventory active/user/approval/provider/child waits, unknown effects, pending child results, unconsumed inputs and consumer lag. At the agreed cutoff explicitly abandon remaining test sessions, retain their unresolved-effect classification, and fence their old workers/callback routes. They are not relabelled quiescent or imported as successful S1 sessions. Preserve required product copies before provisioning a fresh S1 schema/content store/key configuration. Keep any required old audit database readable only with its pinned runtime, outside the S1 pool. No bulk delete or production action is authorized by this document.

.com continues its current non-kernel production path until P2 has passed acceptance. Its first kernel sessions must be S1. Verify this ordering at cutover; if .com has already accumulated pre-S1 kernel data, stop and obtain a migration plan for the W2a categories instead of treating them like disposable .ai data. P3/P4 evolve S1 additively through feature levels, without a second drain.

P2's `export_session` emits versioned JSONL: header `{export_version:1,storage_major:"S1",tenant_id,session_id,head_seq,chain_digest,content_mode}`, ordered reference-form record lines with SQL envelope digests and embedded input proofs, inbox lines with original sequence/received time, admission key/signature/time and consumption state, per-record checkpoints, and a final manifest binding line counts and head. Default `content_mode="skeleton"`; optional authorized content lines carry slot/ref/decoded value and make the export a sensitive independent copy. Capture one consistent head/inbox snapshot and dependency cuts. A forgotten session supports skeleton export only, retaining the original input proofs for verification with retained public keys and no content decryption. Fixtures define exact line discriminators and JCS bytes, including proof encoding. Export is for archive/product history, not a second runtime import format. A later major release runs its export in the old runtime and supplies an offline migration tool creating new sessions with `migrated_from`; unknown stays unknown for every supported migration category.

## Acceptance tests required before P2 completion

These are implementation acceptance cases, not claims that P1 has run them.

| ID | Test and expected evidence |
| --- | --- |
| S1-A1 | Walk canonical record/inbox schemas and producer-specific nested schemas. Every one of the 14 record kinds, nine inbox kinds, seven boundaries, four handle variants, envelope fields and new reserved slots has exactly one disposition. Fail on an unclassified field/table/content slot. Exercise both inbox body and embedded input copies. |
| S1-A2 | Unique markers in prompt, user reply, tool arguments/result, one-byte errors, attributes, seed, model route, approval reason, memory, archive paths and evidence. After execution, scan all SQL tables, logs/exception chains, telemetry, worker temp/debug directories and runtime-owned artifact projections: no plaintext. Explicit body export is the only opted-in exception. Include content shorter than 1 KB. |
| S1-A3 | Destroy/forget a session. Old/new workers and a restored SQL backup cannot unwrap the key or read content. The skeleton chain/checkpoints still verify with no content access. Purge then leaves only complete signed tombstones for root and descendants. Kill after every state-machine boundary; repeated deletion converges and late publication cannot resurrect content. |
| S1-A4 | SQL and S3 content writes cut before upload, after upload, before log commit, after commit/reply loss and during GC. Assert no committed missing object, no live/pinned deletion, deterministic duplicate refs/receipts, bounded orphan cleanup and no committed half-child batch. Expire staging pins and run GC while messages are reachable only inside encrypted request/seed/snapshot containers: flattened links must preserve every needed message, including after inbox consumption and repeated references. Missing object reports ContentUnavailable; tampering reports integrity failure. |
| S1-A5 | Key service unavailable during create, get, put, record/input signing and destroy. No plaintext or unsigned fallback, invented outcome or partial admission. Content key errors defer drive/projection; signed creates, input admissions and appends roll back on signer failure; deletion remains pending. Recovery continues from original identities and satisfies simulation invariants. |
| S1-A6 | Execute every store/content/export/delete method as tenant A against B; close application checks and prove PostgreSQL RLS alone denies. Test absent SET LOCAL, forged A-login/B-context, attempted SET ROLE/SESSION AUTHORIZATION, WITH CHECK writes, TRUNCATE/DDL refusal, owner startup refusal and cross-tenant parent/consumer/commit/content FKs. Reuse an A-authenticated pooled connection across A/B request contexts and prove B remains inaccessible; a separate B login sees only B. Test trusted shared-gateway context separately without claiming credential isolation. Use TCP/password PostgreSQL with separate roles. |
| S1-A7 | Require a signature at every record from creation. Change body/actor, delete/reorder rows and recompute all keyless digests: verification or recovery/extension must reject at the first invalid/missing proof. Seed checkpoint 256 with rewritten unsigned tail 257–260, then attempt honest append/checkpoint: reject before signing; never certify the rewritten history. Test latest signed-head deletion against an independent expected head; absent/stale anchors do not prove latestness. Rotation between bind_key and sign preserves the actual key binding or rolls back. Unsigned local history remains explicitly weaker and cannot be upgraded by retro-signing. |
| S1-A8 | Fork/clone/child delivery re-encrypt content under destination keys. Forget source and read destination; forget destination and read source. Same plaintext in two tenants has unrelated IDs/MACs. Repeated same-session message yields one object per label set and stable stored digest. |
| S1-A9 | Mixed low/high-level workers on real SQLite and PostgreSQL, concurrent push/child delivery and setting changes. No low worker decodes or claims a high session. New worker writes old shape before write_level raises; two-node rolling upgrade with new and ongoing sessions succeeds before/after raising it. Default-field fixtures prove old behavior; malformed unknown fields still fail. |
| S1-A10 | Plaintext low-entropy values cannot be guessed through a retained naked SHA digest, source-derived identity or artifact filename. Reference-form digest fixtures and HMAC vectors pin domains, JCS bytes, framing, AEAD AAD and nonce uniqueness. Include a private low-entropy network policy: its operational hash is absent from clear SQL/logs/skeleton exports, ledger hashes cover encrypted references, and a remote agent without session keys still checks ordinary canonical operational equality. |
| S1-A11 | Existing transaction/recovery tests retain zero partial writes on validation, CAS, expiry and deferred commit failure. Cover both prestaged content and child IDs/content discovered by the original callback; inject timeout/failure during key allocation, upload and signing, then expire the lease before final revalidation. Host business rows, every child and parent admission must all disappear, with only cleanup-owned external orphans allowed. Duplicate callback result+usage compares exact authenticated bytes; late audit usage bills once. Snapshot export replays a complete consistent prefix, including pending inputs. |
| S1-A12 | `scripts/session_kernel_benchmark.py`: same 100-turn / 10 MB-history workload before/after, cold and warm. Record-table bytes and per-step time do not exceed baseline. Report content-store bytes separately and verify message reuse. |
| S1-A13 | .ai drain rehearsal records product-copy completeness, explicit abandoned waits/unknowns, fenced old callbacks and fresh S1 creation. Confirm .com has no pre-S1 kernel rows. This acceptance authorizes no deployment by itself. |
| S1-A14 | Within one tenant, give two authenticated users disjoint session/resource grants. Verify that backend/App Server authorization rejects cross-user reads, inbox/control writes, artifact access, export, deletion and reconnect replay before storage/content access; authorized shared-resource grants still work. Keep worker/service access separate. This proves the user-authorization boundary; tenant RLS alone is not this evidence. |
| S1-A15 | Before consumption, SQL-edit actor/control/approval/request references, target/generation, input_seq or received_ms; recompute body digest or remove/swap the proof. Signed admission verification must reject before decryption, application or new ledger signing. Valid duplicates retain the original proof, sequence and time. Verify copied input proofs in consumed records and skeleton exports after forget with retained public keys; no content key is needed. Cover signer failure, key rotation and complete signed create/push/append atomicity. |

Run focused producer/store tests locally while implementing. The approved million-seed SQLite and 10,000-seed PostgreSQL evidence belongs to the milestone's remote gates; this design requires no local full-suite run.
