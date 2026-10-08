"""The C encoder is guarded by byte equivalence to the full RFC 8785 encoder."""

import base64
import json
import random
from collections import UserDict
from hashlib import sha256
from pathlib import Path

import pytest

from vv_agent.canonical_json import MAX_WIRE_INTEGER, _jcs_encode, _stdlib_compatible, canonical_json_bytes

FIXTURES = Path(__file__).parent / "fixtures" / "parity"


def golden_vectors():
    def walk(value):
        if isinstance(value, dict):
            if "canonical_json_base64" in value or "rfc8785_sha256" in value:
                yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for path in sorted(FIXTURES.glob("*.json")):
        fixture = json.loads(path.read_text(), parse_int=lambda s: int(s) if abs(int(s)) <= MAX_WIRE_INTEGER else float(s))
        for vector in walk(fixture):
            if "scenario_ref" in vector:
                scenario = next(s for s in fixture["scenarios"] if s["id"] == vector["scenario_ref"])
                value = [s for s in scenario["output"]["sections"] if s["stable"]]
            else:
                fields = (
                    "definition",
                    "request",
                    "event",
                    "entry",
                    "value",
                    "payload",
                    "canonical_result",
                    "wire",
                    "request_without_digest",
                    "command_digest_input",
                    "response_digest_input",
                )
                value = next(vector[field] for field in fields if field in vector)
            yield pytest.param(value, vector, id=f"{path.stem}/{vector.get('name', vector.get('scenario_ref', 'golden'))}")


@pytest.mark.parametrize(("value", "vector"), list(golden_vectors()))
def test_all_vendored_jcs_golden_vectors(value, vector):
    reference = _jcs_encode(value).encode("utf-8")
    assert canonical_json_bytes(value) == reference
    if "canonical_json_base64" in vector:
        assert reference == base64.b64decode(vector["canonical_json_base64"])
        assert len(reference) == vector["canonical_json_utf8_bytes"]
    assert sha256(reference).hexdigest() == vector.get("sha256", vector.get("rfc8785_sha256"))


def test_jcs_literal_escape_vector():
    text = "".join(chr(i) for i in range(32)) + '"\\/\u2028\u2029\x7f😀'
    expected = (
        r'"\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\b\t\n\u000b\f\r\u000e\u000f'
        r"\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017\u0018\u0019\u001a\u001b\u001c\u001d\u001e\u001f\"\\/"
        '\u2028\u2029\x7f😀"'
    )
    assert canonical_json_bytes(text) == expected.encode("utf-8")


def test_every_control_escape_del_separators_and_astral_values():
    text = "".join(chr(i) for i in range(0x80)) + "\u2028\u2029😀\uffff"
    value = {"controls": text, '/"\\\x1f': text}
    assert _stdlib_compatible(value)
    assert canonical_json_bytes(value) == _jcs_encode(value).encode()
    assert b"\\u007f" not in canonical_json_bytes(value)
    assert "\u2028\u2029😀".encode() in canonical_json_bytes(value)


@pytest.mark.parametrize("value", [0.0, -0.0, 1e-7, {"😀": 1, "\ue000": 2}, UserDict({"a": 1}), 2**100, -(2**100)])
def test_ineligible_values_use_reference_encoding_or_rejection(value):
    assert not _stdlib_compatible(value)
    try:
        reference = _jcs_encode(value).encode()
    except (ValueError, TypeError, UnicodeError):
        with pytest.raises(ValueError, match="RFC 8785 I-JSON"):
            canonical_json_bytes(value)
    else:
        assert canonical_json_bytes(value) == reference


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff", "\ud800\udfff"])
@pytest.mark.parametrize("location", ["value", "key", "nested"])
def test_fast_path_rejects_all_surrogates(surrogate, location):
    value = {"key": surrogate} if location == "value" else {surrogate: "value"} if location == "key" else [{"x": surrogate}]
    assert not _stdlib_compatible(value)
    with pytest.raises(ValueError, match="RFC 8785 I-JSON"):
        canonical_json_bytes(value)


@pytest.mark.parametrize("eligible_only", [True, False])
def test_generated_nested_values_match_full_encoder(eligible_only):
    rng = random.Random(8785)
    alphabet = [chr(i) for i in range(0x80)] + ["é", "中", "\u2028", "\u2029", "\ue000", "\uffff", "😀", "𐀀"]
    integers = [0, -1, 1, MAX_WIRE_INTEGER, -MAX_WIRE_INTEGER, MAX_WIRE_INTEGER + 1, -(MAX_WIRE_INTEGER + 1), 2**100]

    def text(*, key=False):
        letters = alphabet[:-2] if eligible_only and key else alphabet
        return "".join(rng.choices(letters, k=rng.randrange(8)))

    def generate(depth=0):
        choice = rng.randrange(6 if depth < 4 else 4)
        if choice == 0:
            return rng.choice([None, True, False])
        if choice == 1:
            return rng.choice(integers[:5] if eligible_only else [*integers, 0.0, -0.0, 1e-7, 1e21, 0.75])
        if choice == 2:
            return text()
        if choice == 3:
            return rng.randrange(-MAX_WIRE_INTEGER, MAX_WIRE_INTEGER + 1)
        if choice == 4:
            return [generate(depth + 1) for _ in range(rng.randrange(5))]
        return {text(key=True): generate(depth + 1) for _ in range(rng.randrange(5))}

    for _ in range(3000):
        value = generate()
        if eligible_only:
            assert _stdlib_compatible(value)
        try:
            reference = _jcs_encode(value).encode()
        except (ValueError, TypeError, UnicodeError):
            with pytest.raises(ValueError):
                canonical_json_bytes(value)
        else:
            assert canonical_json_bytes(value) == reference, value
