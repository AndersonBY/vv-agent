# Historical tool receipt

`tool_result_0341.json` was produced by installed `vv-agent==0.34.1` using
`generate_tool_result_0341.py`. It contains the real encoded kernel history and
original provider callback for one 30,000-character result. It contains no
credentials or live provider data. Tests replay the immutable records and restore
the body at the saved artifact path; they do not derive the historical filename.

To regenerate, run the generator with a disposable 0.34.1 Python environment and
make this repository and `tests/` importable (including the test dependencies).
Keep the installed 0.34.1 site-packages first on `PYTHONPATH`, so it supplies
`vv_agent` rather than the current editable installation. Pass the output JSON
path as the sole argument. The generator asserts version 0.34.1 and verifies the
saved archive against the original provider callback before exporting.
