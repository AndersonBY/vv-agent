# Unreleased: artifact reread gaps

This draft has no assigned package version. It requires a new immutable central
contract release before the SDK lock and vendored fixtures can be synchronized.

Compaction evidence can be read and searched in its current session/task. The
same public scope predicate now protects file reads and file metadata. Scope
encoding preserves the full identity instead of losing it through escaping or
truncation. Hosts can call is_task_artifact_path with their trusted host task id,
session id and SDK task id. Other tasks and sessions remain denied.

Search scans long lines in bounded windows and returns line numbers, character
columns and short match excerpts. It reports the overlap limit rather than
silently skipping a line. read_file accepts start_char with fixed 12,000-character pages and reports the
next position; its existing digest-bound cursor remains available. Slow regular
expressions are interrupted and return an incomplete result with earlier hits.
All model-controlled matching shares an interruptible request budget; reverse
mode is rejected. Unicode word and dotless-i changes are listed in upgrading.md.

Command output retention failures preserve command status, exit code and
directive, with an independent unavailable fact. A real shell/kernel regression
checks that driving the completed session again does not repeat the command.

Contract 33.0.0 is a major draft because authorization, search and terminal
failure behavior replace existing behavior. There is no record, inbox, SQL or
App Server schema bump. The published lock and fixture snapshot remain 32.1.1
until an immutable release exists. Local producer checks select the draft via
VV_AGENT_CONTRACT_SOURCE. Guard closure without definitive provider cancellation
preserves the original wait as unconfirmed when the turn is not marked cancelled;
later trusted results are audit-only. User cancel/abort keeps its existing
op_unknown behavior, including provider_result_invalid for invalid Definitive replies.
Frozen turn and old task/compaction read impacts, drain
instructions and new-session authorization are in [upgrading](../upgrading.md).
Frozen Rust remains at contract 23.0.0.
