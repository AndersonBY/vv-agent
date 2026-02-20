"""S3-compatible workspace backend.

Requires ``boto3``.  Install via::

    uv pip install 'v-agent[s3]'

Supports any S3-compatible service (AWS S3, MinIO, Aliyun OSS S3 mode, etc.)
by setting ``endpoint_url``.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import posixpath
from datetime import UTC, datetime
from typing import Any

from v_agent.workspace.base import FileInfo

_BOTO3_AVAILABLE = importlib.util.find_spec("boto3") is not None


def _require_boto3() -> Any:
    if not _BOTO3_AVAILABLE:
        raise ModuleNotFoundError(
            "boto3 is required for S3WorkspaceBackend. "
            "Install it with: uv pip install 'v-agent[s3]'"
        )
    import boto3  # ty: ignore[unresolved-import]
    return boto3


class S3WorkspaceBackend:
    """WorkspaceBackend backed by an S3-compatible bucket.

    Parameters
    ----------
    bucket:
        Bucket name.
    prefix:
        Key prefix for all workspace files (no leading/trailing ``/``).
        Defaults to ``""`` (bucket root).
    endpoint_url:
        Custom endpoint for S3-compatible services (MinIO, OSS, R2, etc.).
        ``None`` uses the default AWS endpoint.
    region_name:
        AWS region. Ignored by most S3-compatible services.
    aws_access_key_id / aws_secret_access_key:
        Credentials. Falls back to the standard boto3 credential chain
        (env vars, ~/.aws/credentials, IAM role) when ``None``.
    addressing_style:
        ``"virtual"`` (default) or ``"path"``.  Most S3-compatible services
        (Aliyun OSS, Cloudflare R2) require virtual-hosted-style.
        MinIO typically needs ``"path"``.
    """

    __slots__ = ("_bucket", "_client", "_prefix")

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        addressing_style: str = "virtual",
    ) -> None:
        boto3 = _require_boto3()
        from botocore.config import Config  # ty: ignore[unresolved-import]

        kwargs: dict[str, Any] = {
            "config": Config(
                s3={
                    "addressing_style": addressing_style,
                    "payload_signing_enabled": False,
                },
                request_checksum_calculation="when_required",
            ),
        }
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if region_name:
            kwargs["region_name"] = region_name
        if aws_access_key_id:
            kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            kwargs["aws_secret_access_key"] = aws_secret_access_key
        self._client: Any = boto3.client("s3", **kwargs)
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    # -- internal helpers ---------------------------------------------------

    def _key(self, path: str) -> str:
        """Map a workspace-relative posix path to an S3 object key."""
        norm = posixpath.normpath(path).lstrip("/")
        if norm == ".":
            norm = ""
        if self._prefix:
            return f"{self._prefix}/{norm}" if norm else self._prefix
        return norm

    def _rel(self, key: str) -> str:
        """Strip the prefix from an S3 key to get the workspace-relative path."""
        if self._prefix and key.startswith(self._prefix + "/"):
            return key[len(self._prefix) + 1:]
        return key

    def _list_keys(self, prefix: str) -> list[str]:
        """List all object keys under *prefix* (handles pagination)."""
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    # -- Protocol implementation --------------------------------------------

    def list_files(self, base: str, glob: str) -> list[str]:
        base_norm = posixpath.normpath(base).lstrip("/")
        if base_norm == ".":
            search_prefix = self._prefix
        elif self._prefix:
            search_prefix = f"{self._prefix}/{base_norm}"
        else:
            search_prefix = base_norm

        if search_prefix and not search_prefix.endswith("/"):
            search_prefix += "/"

        all_keys = self._list_keys(search_prefix)

        pattern = posixpath.join(base_norm, glob) if base_norm and base_norm != "." else glob
        result: list[str] = []
        for key in all_keys:
            rel = self._rel(key)
            if not rel or rel.endswith("/"):
                continue
            if fnmatch.fnmatch(rel, pattern) or self._glob_match(rel, pattern):
                result.append(rel)
        result.sort()
        return result

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        resp = self._client.get_object(Bucket=self._bucket, Key=self._key(path))
        return resp["Body"].read()

    def write_text(self, path: str, content: str, *, append: bool = False) -> int:
        key = self._key(path)
        if append:
            try:
                existing = self.read_text(path)
            except self._client.exceptions.NoSuchKey:
                existing = ""
            content = existing + content
        data = content.encode("utf-8")
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentLength=len(data),
        )
        return len(content)

    def file_info(self, path: str) -> FileInfo | None:
        key = self._key(path)
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
        except self._client.exceptions.ClientError:
            return None
        modified = head.get("LastModified")
        modified_at = modified.isoformat() if modified else datetime.now(tz=UTC).isoformat()
        suffix = ""
        dot = path.rfind(".")
        if dot != -1 and "/" not in path[dot:]:
            suffix = path[dot:]
        return FileInfo(
            path=self._rel(key),
            is_file=True,
            is_dir=False,
            size=head.get("ContentLength", 0),
            modified_at=modified_at,
            suffix=suffix,
        )

    def exists(self, path: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._key(path))
            return True
        except self._client.exceptions.ClientError:
            return False

    def is_file(self, path: str) -> bool:
        return self.exists(path)

    def mkdir(self, path: str) -> None:
        # S3 没有真正的目录, 无需操作
        pass

    @staticmethod
    def _glob_match(path: str, pattern: str) -> bool:
        """支持 ``**`` 的 glob 匹配."""
        import re as _re

        parts: list[str] = []
        i = 0
        while i < len(pattern):
            if pattern[i:i + 3] == "**/":
                parts.append("(?:.+/)?")
                i += 3
            elif pattern[i:i + 2] == "**":
                parts.append(".*")
                i += 2
            elif pattern[i] == "*":
                parts.append("[^/]*")
                i += 1
            elif pattern[i] == "?":
                parts.append("[^/]")
                i += 1
            else:
                parts.append(_re.escape(pattern[i]))
                i += 1
        regex = "^" + "".join(parts) + "$"
        return _re.match(regex, path) is not None
