from __future__ import annotations

from typing import cast

import pytest

from vv_agent.kernel._sql import LedgerIntegrityError, _digest_bytes


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param(b"0" * 64, id="bytes"),
        pytest.param(0, id="integer"),
        pytest.param("", id="empty"),
        pytest.param("0" * 63, id="too-short"),
        pytest.param("0" * 65, id="too-long"),
        pytest.param("A" + "0" * 63, id="uppercase"),
        pytest.param("g" + "0" * 63, id="not-hex"),
        pytest.param("0" * 63 + "\n", id="trailing-newline"),
        pytest.param("0" * 32 + "\n" + "0" * 31, id="embedded-newline"),
        pytest.param("0" * 32 + " " + "0" * 31, id="embedded-space"),
        pytest.param("0" * 63 + " ", id="trailing-space"),
        pytest.param("0" * 63 + "\u00e9", id="non-ascii"),
        pytest.param("\ud800" * 64, id="surrogate"),
    ],
)
def test_digest_bytes_rejects_non_lowercase_sha256(value: object):
    with pytest.raises(LedgerIntegrityError) as raised:
        _digest_bytes(cast(str, value), "record digest")

    assert str(raised.value) == "record digest must be a lowercase SHA-256 digest"


@pytest.mark.parametrize(
    "value",
    ["0" * 64, "0123456789abcdef" * 4],
    ids=["all-zero", "all-hex-digits"],
)
def test_digest_bytes_accepts_lowercase_sha256(value: str):
    assert _digest_bytes(value, "record digest") == bytes.fromhex(value)
