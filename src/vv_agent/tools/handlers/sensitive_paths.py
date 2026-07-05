from __future__ import annotations

from pathlib import PurePosixPath

_SENSITIVE_EXACT_NAMES = {
    ".env",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

_SENSITIVE_SUFFIXES = {
    ".key",
    ".pem",
    ".p8",
    ".p12",
    ".pfx",
}

_SENSITIVE_NAME_TOKENS = (
    "credential",
    "credentials",
    "secret",
    "secrets",
    "token",
    "private_key",
)


def is_sensitive_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        return False
    parts = PurePosixPath(normalized).parts
    name = parts[-1].lower()

    if name in _SENSITIVE_EXACT_NAMES:
        return True
    if name.startswith(".env.") and name not in {".env.example", ".env.sample", ".env.template"}:
        return True
    if name.startswith(("secrets.", "secret.")):
        return True
    if any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES):
        return True
    if name.endswith(".env"):
        return True
    if any(token in name for token in _SENSITIVE_NAME_TOKENS):
        config_dirs = {".config", "config", "configs", "keys", "secrets", ".ssh", ".aws", ".gcp"}
        return any(part.lower() in config_dirs for part in parts[:-1])
    return False
