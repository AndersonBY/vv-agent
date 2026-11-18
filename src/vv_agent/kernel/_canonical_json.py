"""RFC 8785 canonical JSON encoding and SHA-256 validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from json.encoder import encode_basestring
from typing import Any

MAX_WIRE_INTEGER = (1 << 53) - 1


_ASTRAL_KEY_RE = re.compile("[\ud800-\udfff\U00010000-\U0010ffff]")
_SURROGATE_RE = re.compile("[\ud800-\udfff]")


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any, field_name: str = "value") -> bytes:
    return _canonical_json(value, field_name).encode("utf-8")


def _canonical_json(value: Any, field_name: str) -> str:
    try:
        if _stdlib_compatible(value):
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return _jcs_encode(value, fast=True)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{field_name} must be RFC 8785 I-JSON: {exc}") from exc


def canonical_json_sha256(value: Any, field_name: str = "value") -> str:
    return hashlib.sha256(canonical_json_bytes(value, field_name)).hexdigest()


def validate_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _stdlib_compatible(value: Any) -> bool:
    # BMP keys sort identically by code point and UTF-16; C string escapes match JCS.
    kind = type(value)
    if kind is str:
        return value.isascii() or _SURROGATE_RE.search(value) is None
    if value is None or kind is bool:
        return True
    if kind is int:
        return -MAX_WIRE_INTEGER <= value <= MAX_WIRE_INTEGER
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str or (not key.isascii() and _ASTRAL_KEY_RE.search(key) is not None):
                return False
            if not _stdlib_compatible(item):
                return False
        return True
    if kind is list or kind is tuple:
        return all(_stdlib_compatible(item) for item in value)
    return False


def _jcs_encode(value: Any, *, fast: bool = False) -> str:
    if fast and isinstance(value, dict | list | tuple) and _stdlib_compatible(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_quote(value)
    if isinstance(value, int):
        if not -MAX_WIRE_INTEGER <= value <= MAX_WIRE_INTEGER:
            raise ValueError("integer is outside the I-JSON safe range")
        return str(value)
    if isinstance(value, float):
        return _jcs_float(value)
    if isinstance(value, Mapping):
        items: list[str] = []
        keys: list[str] = []
        for key in value:
            if not isinstance(key, str):
                raise TypeError("object keys must be strings")
            keys.append(key)
        for key in sorted(keys, key=utf16_sort_key):
            items.append(f"{_jcs_quote(key)}:{_jcs_encode(value[key], fast=fast)}")
        return "{" + ",".join(items) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_jcs_encode(item, fast=fast) for item in value) + "]"
    raise TypeError(f"unsupported JSON value type {type(value).__name__}")


def _jcs_quote(value: str) -> str:
    # The stdlib C encoder uses JCS string escapes; reject lone surrogates first.
    value.encode("utf-8")
    return encode_basestring(value)


def _jcs_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    if value == 0:
        return "0"
    negative = value < 0
    absolute = -value if negative else value
    source = repr(absolute).lower()
    if "e" in source:
        mantissa, exponent_text = source.split("e", 1)
        exponent = int(exponent_text)
        digits = mantissa.replace(".", "").rstrip("0")
        digits = digits or "0"
    else:
        exponent = 0
        digits = source

    if 1e-6 <= absolute < 1e21:
        if "e" in source:
            decimal_at = exponent + 1
            if decimal_at <= 0:
                rendered = "0." + ("0" * -decimal_at) + digits
            elif decimal_at >= len(digits):
                rendered = digits + ("0" * (decimal_at - len(digits)))
            else:
                rendered = digits[:decimal_at] + "." + digits[decimal_at:]
        else:
            rendered = source.removesuffix(".0")
    else:
        if "e" not in source:
            integer, _, fraction = source.partition(".")
            all_digits = (integer + fraction).lstrip("0")
            first_index = next(index for index, char in enumerate(source) if char not in "0.")
            dot_index = source.find(".")
            exponent = (dot_index if dot_index >= 0 else len(source)) - first_index - 1
            digits = all_digits.rstrip("0")
        mantissa = digits[0]
        if len(digits) > 1:
            mantissa += "." + digits[1:]
        sign = "+" if exponent >= 0 else ""
        rendered = f"{mantissa}e{sign}{exponent}"
    return "-" + rendered if negative else rendered


def utf16_sort_key(value: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise TypeError("object keys must be strings")
    _jcs_quote(value)
    return tuple(value.encode("utf-16-be"))
