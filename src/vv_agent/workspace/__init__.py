from vv_agent.workspace.base import (
    INVALID_EXCLUDE_FILES_PATTERN_CODE,
    INVALID_EXCLUDE_FILES_PATTERN_MESSAGE,
    DiscoveryFilteredWorkspaceBackend,
    FileInfo,
    InvalidPortableRegexError,
    WorkspaceBackend,
    compile_portable_workspace_regex,
)
from vv_agent.workspace.local import LocalWorkspaceBackend
from vv_agent.workspace.memory import MemoryWorkspaceBackend

__all__ = [
    "INVALID_EXCLUDE_FILES_PATTERN_CODE",
    "INVALID_EXCLUDE_FILES_PATTERN_MESSAGE",
    "DiscoveryFilteredWorkspaceBackend",
    "FileInfo",
    "InvalidPortableRegexError",
    "LocalWorkspaceBackend",
    "MemoryWorkspaceBackend",
    "S3WorkspaceBackend",
    "WorkspaceBackend",
    "compile_portable_workspace_regex",
]


def __getattr__(name: str) -> type:
    if name == "S3WorkspaceBackend":
        from vv_agent.workspace.s3 import S3WorkspaceBackend
        return S3WorkspaceBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
