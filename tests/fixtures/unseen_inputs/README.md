# Input projection regression records

`minimal-False.json` and `minimal-True.json` retain the two ten-record review
counterexamples: a tool dispatch, an unchanged or prepended before-memory
snapshot, an authenticated callback input and its normal receipt. The prepend
case includes the final receipt from the same source sequence so both cases
check the adopted result as well as the callback's protection position.

`producer-corpus.json.gz` contains 24 real Python producer logs selected from
the contract 37 input-protection review corpus. Each entry retains its producer
test name and complete encoded records. The gzip timestamp is zero. It covers
native child notifications, user and tool replies, approval denial/timeout,
provider/model callbacks, corrections/audits, in-flight steer, after-cycle
steering, summary/micro/emergency replacement, HTTP PTL/retry/Unknown and restart.
Native child correction is not a reachable scenario in these logs.

`tests/session/input_projection_reference.py` reconstructs each expected prefix
from record zero without using the incremental protection or boundary helpers.
The differential suite also inserts fixed-seed in-flight snapshots into these
producer sequences, remapping captured record versions after insertion.
