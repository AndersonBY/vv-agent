from vv_agent.tools.handlers.sensitive_paths import is_sensitive_path


def test_sensitive_path_classifier_flags_common_secret_paths() -> None:
    sensitive = [
        ".env",
        ".env.local",
        "keys/AuthKey_ABC123.p8",
        "keys/private.key",
        ".ssh/id_rsa",
        "config/service_token.json",
        "secrets.production",
        ".npmrc",
    ]
    for path in sensitive:
        assert is_sensitive_path(path), path


def test_sensitive_path_classifier_allows_normal_project_files() -> None:
    safe = [
        ".env.example",
        "src/tokenizer.py",
        "docs/secrets-management.md",
        "config/example.json",
    ]
    for path in safe:
        assert not is_sensitive_path(path), path
