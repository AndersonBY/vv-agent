from vv_agent.runtime.backends.base import ExecutionBackend
from vv_agent.runtime.backends.inline import InlineBackend
from vv_agent.runtime.backends.thread import ThreadBackend

_DISTRIBUTED_EXPORTS = {
    "DistributedAdvanceDecision",
    "DistributedDeliveryOutcome",
    "DistributedRunHandle",
    "DistributedWaitReason",
}

__all__ = [
    "DistributedAdvanceDecision",
    "DistributedDeliveryOutcome",
    "DistributedRunHandle",
    "DistributedWaitReason",
    "ExecutionBackend",
    "InlineBackend",
    "ThreadBackend",
]


def __getattr__(name: str):
    if name not in _DISTRIBUTED_EXPORTS:
        raise AttributeError(name)
    from vv_agent.runtime.backends import distributed

    return getattr(distributed, name)
