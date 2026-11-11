# Stored 0.34.1 tool receipt

`tool_result_0341.json` is the original 0.34.1 long provider receipt from WIP
commit `38d9b89dbe738a344448f4a5865afa1b2c068021`. It contains the original archive
reference and unbounded callback. `test_legacy_tool_receipts.py` imports its full
record sequence and checks duplicate receipt validation and immutable resources.

Run `generate_tool_result_0341.py` only with an unmodified 0.34.1 source tree and
its environment. Regenerating it with the new archive identity would replace
the historical evidence.
