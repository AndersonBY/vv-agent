from vv_agent.workspace.base import FileInfo, WorkspaceBackend
from vv_agent.workspace.local import LocalWorkspaceBackend
from vv_agent.workspace.memory import MemoryWorkspaceBackend

__all__ = [
    "FileInfo",
    "LocalWorkspaceBackend",
    "MemoryWorkspaceBackend",
    "S3WorkspaceBackend",
    "WorkspaceBackend",
]


def __getattr__(name: str) -> type:
    if name == "S3WorkspaceBackend":
        from vv_agent.workspace.s3 import S3WorkspaceBackend
        return S3WorkspaceBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
