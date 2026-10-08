"""The optimized string encoder must retain JCS bytes and rejection boundaries."""

import pytest

from vv_agent.canonical_json import canonical_json_bytes


def test_jcs_string_escapes_and_utf16_key_order():
    assert canonical_json_bytes('"\\\b\t\n\f\r\x00\x1f/\u2028😀') == '"\\"\\\\\\b\\t\\n\\f\\r\\u0000\\u001f/\u2028😀"'.encode()
    assert canonical_json_bytes({"\ue000": 1, "😀": 2}) == '{"😀":2,"\ue000":1}'.encode()


@pytest.mark.parametrize("value", ["\ud800", "\udfff", {"\ud800": 1}, ["\ud800\udfff"]])
def test_jcs_rejects_surrogates(value):
    with pytest.raises(ValueError):
        canonical_json_bytes(value)
