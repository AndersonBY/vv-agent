# Changelog

## 0.33.1 — 2026-11-05

Fix the 0.33.0 search CPU budget design error that stopped ordinary large-file
regex/literal scans before late matches. Each engine invocation now has its own
250 ms ReDoS timeout; enumeration and matching share a 10 s thread CPU limit.
Earlier hits and stop coordinates survive either limit, with distinct messages.
Local, Memory, S3, artifacts, globs/find_files and child exclusions use one rule.
Existing window, page, head and file limits remain unchanged.

Requires contract 34.0.0 under its observable behavior version policy. This is
handler-only: tool schemas, frozen definitions and all store/wire versions stay
unchanged; 0.33.0 turns need no special transition. See the
[release notes](docs/releases/0.33.1.md) and [upgrade guide](docs/upgrading.md).

## 0.33.0 — 2026-11-04

Requires contract 33.0.0. read_file adds character positioning, removes max_chars
and uses fixed 12,000-character pages. Search reads long lines with bounded
windows and interrupts slow matching. Artifact authorization now preserves full,
type-separated task and session identities.

Old executable turns close at the definition guard after the read_file schema
change. Old nonliteral task directories and lossy compaction paths lose model
read authorization. Terminal output retention failures preserve confirmed command
results with retention unavailable and no recovery pointer. Guard closure keeps
unconfirmed provider waits when the turn is not marked cancelled; late results
are audit-only. User cancel/abort retains its existing behavior.

No record, inbox, SQL or App Server schema bump. Read the
[release notes](docs/releases/0.33.0.md) and [upgrade guide](docs/upgrading.md)
before switching workers. Earlier release notes are in [docs/releases/](docs/releases/).
