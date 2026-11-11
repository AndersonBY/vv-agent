"""Synchronous transaction callback contract; no arity fallback or ambient tx."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from functools import partial
from threading import local
from typing import Any, Protocol

from .store import SessionTx

_state = local()


class TransactionHook(Protocol):
    def __call__(self, point: str, record: Any, *, tx: SessionTx | None) -> None: ...


class Admission(Protocol):
    def __call__(self, plan: Any, *, tx: SessionTx) -> Any: ...


class SummaryAdmission(Protocol):
    def __call__(self, plan: Any, summary: str | None, *, tx: SessionTx) -> Any: ...


class CallbackContractError(TypeError):
    code = "transaction_callback_contract"

    def __init__(self):
        super().__init__(self.code)


def validate(callback, arity: int):
    original = getattr(callback, "_transaction_original", None)
    if original is not None:
        validate(original, arity)
    try:
        signature = inspect.signature(callback)
        tx = signature.parameters["tx"]
        if tx.kind is not inspect.Parameter.KEYWORD_ONLY or tx.default is not inspect.Parameter.empty:
            raise CallbackContractError()
        signature.bind(*([None] * arity), tx=None)
    except (TypeError, ValueError, KeyError) as exc:
        raise CallbackContractError() from exc
    target = callback
    while isinstance(target, partial):
        target = target.func
    for candidate in (callback, target):
        if inspect.iscoroutinefunction(candidate) or inspect.iscoroutinefunction(type(candidate).__call__):
            raise CallbackContractError()
    return callback


def synchronous(value):
    if inspect.isawaitable(value):
        error = CallbackContractError()
        if inspect.iscoroutine(value) or inspect.isgenerator(value):
            try:
                value.close()
            except BaseException as cleanup:
                error.add_note(f"coroutine close failed: {type(cleanup).__name__}")
        raise error
    return value


@contextmanager
def transaction_callback():
    previous = getattr(_state, "transaction", False)
    _state.transaction = True
    try:
        yield
    finally:
        _state.transaction = previous


def execution_guard():
    if getattr(_state, "transaction", False):
        from .ownership import OwnershipError

        raise OwnershipError("transaction_callback_reentry")


def hook(runtime, point, record, *, tx):
    callback = validate(runtime.hook, 2)
    if tx is not None:
        with transaction_callback():
            synchronous(callback(point, record, tx=tx))
    else:
        synchronous(callback(point, record, tx=None))


def validate_runtime(runtime):
    from .children import ChildTool, SummaryChildTool

    validate(runtime.hook, 2)
    for binding in tuple(runtime.children.values()):
        validate(
            binding.admit if isinstance(binding, (ChildTool, SummaryChildTool)) else binding,
            2 if isinstance(binding, SummaryChildTool) else 1,
        )


def noop_hook(point, record, *, tx):
    pass
