"""Public microcompaction policy bindings to the kernel-owned values."""

from vv_agent.kernel._microcompaction_policy import MicrocompactionPolicy as MicrocompactionPolicy
from vv_agent.kernel._microcompaction_policy import normalize_microcompaction_policy as normalize_microcompaction_policy

__all__ = [
    "MicrocompactionPolicy",
    "normalize_microcompaction_policy",
]
