"""Explicit checkout ownership. The ledger never routes SQL or holds its lock over I/O."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, get_ident, local
from weakref import ref


class OwnershipError(RuntimeError):
    def __init__(self, code="connection_ownership", *, stage="checkout"):
        self.code, self.stage = code, stage
        super().__init__(f"{code}: {stage}")


_lock = Lock()
_lock_owner: int | None = None
_ledger_local = local()


def _check_lock_reentry():
    if getattr(_ledger_local, "entered", False):
        raise OwnershipError(stage="lock-reentry")


@contextmanager
def _ledger_lock():
    global _lock_owner
    _check_lock_reentry()
    _ledger_local.entered = True
    try:
        with _lock:
            _lock_owner = get_ident()
            try:
                yield
            finally:
                _lock_owner = None
    finally:
        _ledger_local.entered = False


class _HostOwners:
    def __init__(self, references):
        self.references = references

    def __call__(self):
        for reference in reversed(self.references):
            if (store := reference()) is not None:
                return store
        return None


_hosts: dict[object, tuple[_HostOwners, int]] = {}
_active: dict[object, tuple[object, object]] = {}
_opening_threads: dict[int, int] = {}


def _native_identity(store):
    connection = store.connection
    if hasattr(connection, "pgconn"):
        raw = connection.pgconn
        return (("pgraw", id(raw)),), raw
    return (("sqlite", id(connection)),), connection


def native_keys(store):
    return _native_identity(store)[0]


def _host_alive(host):
    store = host[0]() if host else None
    return store is not None and not getattr(store, "_checkout_closed", False) and not getattr(store.connection, "closed", False)


def forget_host(store):
    with _ledger_lock():
        for key in getattr(store, "_ownership_keys", ()):
            current = _hosts.get(key)
            if current is not None and current[0]() is store:
                del _hosts[key]


def register_host(store, *, input_only=False):
    _check_lock_reentry()
    keys = native_keys(store)
    store._ownership_keys = keys
    thread = get_ident()
    with _ledger_lock():
        for key in keys:
            host = _hosts.get(key)
            if key in _active or (
                host is not None
                and _host_alive(host)
                and (host[1] != thread or (_opening_threads.get(thread, 0) and not input_only))
            ):
                raise OwnershipError(stage="initialization")
        previous = {key: _hosts.get(key) for key in keys}
        if not input_only:
            for key in keys:
                old = _hosts.get(key)
                references = tuple(r for r in old[0].references if r() is not None) if old is not None else ()
                _hosts[key] = (_HostOwners((*references, ref(store))), thread)
        return previous


def restore_host(store, previous):
    with _ledger_lock():
        for key, old in previous.items():
            current = _hosts.get(key)
            if current is not None and current[0]() is store:
                if _host_alive(old):
                    _hosts[key] = old
                else:
                    del _hosts[key]


@contextmanager
def preserving_exit(factory, exit_errors=None):
    """A cleanup exception is secondary to the exact original execution error."""
    _check_lock_reentry()
    thread = get_ident()
    with _ledger_lock():
        _opening_threads[thread] = _opening_threads.get(thread, 0) + 1
    try:
        context = factory()
        if not callable(getattr(context, "__enter__", None)) or not callable(getattr(context, "__exit__", None)):
            raise OwnershipError(stage="factory-context")
        value = context.__enter__()
    finally:
        with _ledger_lock():
            _opening_threads[thread] -= 1
            if not _opening_threads[thread]:
                del _opening_threads[thread]
    try:
        yield value
    except BaseException as error:
        try:
            context.__exit__(type(error), error, error.__traceback__)
        except BaseException as cleanup:
            if exit_errors is not None:
                exit_errors.append(cleanup)
            error.add_note(f"checkout exit failed: {type(cleanup).__name__}")
        raise
    else:
        try:
            context.__exit__(None, None, None)
        except BaseException as cleanup:
            if exit_errors is not None:
                exit_errors.append(cleanup)
            raise


@contextmanager
def checkout(source, *, factory=None, role="execution"):
    _check_lock_reentry()
    from .callbacks import execution_guard
    from .sql import SQLStore

    execution_guard()
    if hasattr(source, "_committed_guard"):
        source._committed_guard()
    factory = factory if factory is not None else getattr(source, "execution_store", None)
    if not callable(factory):
        raise OwnershipError(stage=role)
    with _ledger_lock():
        # Pin raw identities across factory entry, where another H may close.
        known_hosts = {
            key: owner.connection
            for key, host in _hosts.items()
            for owner in (host[0](),)
            if owner is not None and _host_alive(host)
        }
    token = object()
    keys = ()
    candidate = None
    exit_errors = []
    try:
        with preserving_exit(factory, exit_errors) as candidate:
            # Opaque adapters own the pool's identity proof. No adapter registry.
            if not isinstance(candidate, SQLStore):
                if any(
                    not callable(getattr(candidate, name, None))
                    for name in ("atomic", "read_state", "acquire", "renew", "release")
                ):
                    raise OwnershipError(stage="factory-store")
                yield candidate
                return
            memory = getattr(source, "path", None) == ":memory:"
            if memory:
                if candidate is not source:
                    raise OwnershipError(stage=role)
                candidate._committed_guard()
                yield candidate
                return
            if getattr(candidate.connection, "closed", False):
                raise OwnershipError(stage="closed")
            proposed, raw = _native_identity(candidate)
            source_keys = getattr(source, "_ownership_keys", ())
            with _ledger_lock():
                for key in proposed:
                    host = _hosts.get(key)
                    if (
                        key in _active
                        or key in source_keys
                        or key in known_hosts
                        or (host is not None and _host_alive(host) and host[0]() is not candidate)
                    ):
                        raise OwnershipError(stage=role)
                keys = proposed
                for key in keys:
                    _active[key] = (token, raw)
            # An opaque adapter owns the cross-checkout target guarantee. We
            # still validate every observable SQL candidate and known raw alias.
            candidate._validate_checkout(source if isinstance(source, SQLStore) else candidate)
            yield candidate
    finally:
        # Keep the reservation across factory exit. A failed exit with an open
        # handle remains quarantined; never roll back/close another owner here.
        closed = getattr(getattr(candidate, "connection", None), "closed", False)
        closed = closed or getattr(candidate, "_checkout_closed", False)
        if not exit_errors or closed:
            with _ledger_lock():
                for key in keys:
                    entry = _active.get(key)
                    if entry is not None and entry[0] is token:
                        del _active[key]
                    host = _hosts.get(key)
                    if key not in known_hosts and host is not None and host[0]() is candidate:
                        del _hosts[key]
        # A retained error traceback must not extend these raw lifetimes.
        known_hosts.clear()


def committed_guard(store):
    from .callbacks import execution_guard

    execution_guard()
    guard = getattr(store, "_committed_guard", None)
    if callable(guard):
        guard()
