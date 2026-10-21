"""Client identities cannot overlap the kernel's slash-delimited namespace."""

import re

CLIENT_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}(?![\s\S])"


def client_id(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(CLIENT_ID_PATTERN, value) is None:
        raise ValueError(f"{name} must match [A-Za-z0-9._:-] and contain 1..128 characters")
    return value
