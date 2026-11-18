"""Bounded local session worker lifecycle and observation command queue."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Condition, Event, Thread, current_thread
from time import monotonic
from typing import Protocol
from uuid import uuid4

from vv_agent.kernel._store import ProjectionConsumers, WorkKind

from .store import Lease, LeaseLost, SessionStore
from .supervisor import tick


class DriveCallback(Protocol):
    def __call__(self, store: SessionStore, session_id: str, *, lease: Lease) -> None: ...


ObservationCallback = Callable[[SessionStore], None]


@dataclass(frozen=True, slots=True)
class WorkerSupport:
    """Store and callbacks owned by one :class:`LocalWorker` lifecycle.

    The coordinator owns the store used for observation and projection. Each
    drive opens a fresh store on its dedicated thread. Once ``drive`` is
    invoked, the callback owns the claimed lease lifecycle, including errors.
    """

    open_store: Callable[[], AbstractContextManager[SessionStore]]
    drive: DriveCallback | None = None
    project: Callable[[SessionStore, str, str], None] | None = None
    observe: ObservationCallback | None = None
    projection_consumers: ProjectionConsumers | None = None
    page_size: int = 100
    failure_backoff_ms: int = 1000
    lease_ttl_ms: int = 60_000
    level: int = 1
    handler_versions: tuple[str, ...] = ("1",)
    capabilities: tuple[str, ...] = ()
    all_tenants: bool = False


@dataclass(frozen=True, slots=True)
class _DriveCompletion:
    """Return an unhanded claim to the coordinator or report callback failure."""

    unhanded_lease: Lease | None = None
    error: Exception | None = None


class LocalWorker:
    """Run one coordinator thread and one bounded drive slot."""

    def __init__(
        self,
        support: WorkerSupport,
        *,
        scan_interval: float = 0.25,
        name: str = "session-local-worker",
    ) -> None:
        if isinstance(scan_interval, bool) or not isinstance(scan_interval, (int, float)) or scan_interval <= 0:
            raise ValueError("scan_interval must be positive")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self.support = support
        self.scan_interval = float(scan_interval)
        self._wake = Event()
        self._stop = Event()
        self._state = Condition()
        self._commands: Queue[ObservationCallback] = Queue()
        self._pending_observations = 0
        self._executor: ThreadPoolExecutor | None = None
        self._inflight: Future[_DriveCompletion] | None = None
        self._inflight_lease: Lease | None = None
        self._drive_thread_ident: int | None = None
        self._thread: Thread | None = None
        self._last_error: Exception | None = None
        self._name = name
        self._owner = f"local-worker/{uuid4().hex}"

    @property
    def started(self) -> bool:
        with self._state:
            thread = self._thread
            return thread is not None and thread.ident is not None

    @property
    def running(self) -> bool:
        with self._state:
            thread = self._thread
            return thread is not None and thread.is_alive() and not self._stop.is_set()

    @property
    def last_error(self) -> Exception | None:
        with self._state:
            return self._last_error

    def start(self) -> None:
        with self._state:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                return
            raise RuntimeError("LocalWorker cannot be restarted")
        if self._stop.is_set():
            raise RuntimeError("LocalWorker is stopped")
        self._ensure_executor_locked()
        self._thread = Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        self._wake.set()

    def _ensure_executor_locked(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"{self._name}-drive",
            )
        return self._executor

    def wake(self, session_id: str) -> None:
        """Wake the coordinator; ``session_id`` is only a hint and may be dropped."""

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        self._wake.set()

    def post_observation(self, callback: ObservationCallback) -> None:
        """Submit a short metadata operation to the coordinator thread.

        The callback receives the coordinator's own store. Submission starts the
        worker when needed. The callback must not drive a session or wait for a
        host transaction held by its caller.
        """

        if not callable(callback):
            raise TypeError("observation callback must be callable")
        with self._state:
            if self._stop.is_set():
                raise RuntimeError("LocalWorker is stopped")
            if self._thread is not None and not self._thread.is_alive():
                raise RuntimeError("LocalWorker cannot be restarted")
            self._start_locked()
            self._pending_observations += 1
            self._commands.put(callback)
        self._wake.set()

    def wait_idle(self, timeout: float | None = None) -> None:
        """Wait for local observation commands and the current drive job only."""

        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0):
            raise ValueError("timeout must be a nonnegative number or None")
        deadline = None if timeout is None else monotonic() + float(timeout)
        with self._state:
            if self._thread is not None and current_thread() is self._thread:
                raise RuntimeError("LocalWorker coordinator cannot wait for itself")
            if self._drive_thread_ident == current_thread().ident:
                raise RuntimeError("LocalWorker drive cannot wait for itself")
            while self._pending_observations or self._inflight is not None:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("LocalWorker did not become idle before timeout")
                self._state.wait(remaining)

    def stop(self) -> None:
        self._stop.set()
        with self._state:
            inflight = self._inflight
            if inflight is not None:
                inflight.cancel()
            self._state.notify_all()
        self._wake.set()

    def join(self, timeout: float | None = None) -> bool:
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0):
            raise ValueError("timeout must be a nonnegative number or None")
        with self._state:
            thread = self._thread
            drive_ident = self._drive_thread_ident
        if thread is None:
            return True
        if current_thread() is thread:
            raise RuntimeError("LocalWorker cannot join itself")
        if drive_ident == current_thread().ident:
            raise RuntimeError("LocalWorker drive cannot join its coordinator")
        thread.join(timeout)
        return not thread.is_alive()

    def close(self, timeout: float | None = None) -> bool:
        self.stop()
        return self.join(timeout)

    def __enter__(self) -> LocalWorker:
        self.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        if not self.close():
            raise RuntimeError("LocalWorker did not stop before close returned")

    def _configure_store(
        self,
        store: SessionStore,
        *,
        all_tenants: bool = False,
        support: WorkerSupport | None = None,
    ) -> None:
        selected = self.support if support is None else support
        configure = getattr(store, "configure_worker", None)
        if callable(configure):
            configure(
                level=selected.level,
                handler_versions=selected.handler_versions,
                capabilities=selected.capabilities,
                all_tenants=all_tenants,
            )

    def _open_coordinator_store(self) -> AbstractContextManager[SessionStore]:
        return self.support.open_store()

    def _defer_release_unhanded(self, store: SessionStore, lease: Lease) -> None:
        """Return a claim exactly once when ``drive`` never received ownership."""

        bound = store.for_tenant(lease.tenant_id)
        try:
            bound.defer_drive(lease, retry_after_ms=self.support.failure_backoff_ms)
        except LeaseLost:
            pass
        finally:
            bound.release(lease)

    @staticmethod
    def _combine_errors(*errors: Exception | None) -> Exception | None:
        selected = [error for error in errors if error is not None]
        if not selected:
            return None
        if len(selected) == 1:
            return selected[0]
        return ExceptionGroup("LocalWorker failures", selected)

    def _record_error(self, error: Exception | None) -> None:
        if error is None:
            return
        with self._state:
            self._last_error = self._combine_errors(self._last_error, error)

    def _drain_observations(self, store: SessionStore) -> Exception | None:
        error = None
        while True:
            try:
                callback = self._commands.get_nowait()
            except Empty:
                break
            try:
                callback(store)
            except Exception as exc:
                error = self._combine_errors(error, exc)
            finally:
                with self._state:
                    self._pending_observations -= 1
                    self._state.notify_all()
                self._commands.task_done()
        return error

    def _fail_observations(self, error: Exception) -> int:
        """Fail accepted commands when the coordinator can no longer open a store."""

        discarded = 0
        while True:
            try:
                self._commands.get_nowait()
            except Empty:
                break
            self._commands.task_done()
            with self._state:
                if self._pending_observations:
                    self._pending_observations -= 1
                self._state.notify_all()
            discarded += 1
        if discarded:
            self._record_error(error)
        return discarded

    def _future_done(self, _future: Future[_DriveCompletion]) -> None:
        with self._state:
            self._state.notify_all()
        self._wake.set()

    def _reap_inflight(self, store: SessionStore) -> Exception | None:
        with self._state:
            future = self._inflight
            lease = self._inflight_lease
            if future is None or not future.done():
                return None

        completion: _DriveCompletion | None = None
        error: Exception | None = None
        cleanup_lease = future.cancelled()
        try:
            if not future.cancelled():
                completion = future.result()
                if completion.unhanded_lease is not None:
                    cleanup_lease = True
                error = completion.error
        except Exception as exc:
            error = exc

        if cleanup_lease and lease is not None:
            try:
                self._defer_release_unhanded(store, lease)
            except Exception as exc:
                error = self._combine_errors(error, exc)

        with self._state:
            if self._inflight is future:
                self._inflight = None
                self._inflight_lease = None
            self._state.notify_all()
        return error

    def _periodic_scan(self, store: SessionStore) -> Exception | None:
        error: Exception | None = None
        support = self.support
        self._configure_store(store, all_tenants=support.all_tenants, support=support)
        if support.observe is not None:
            try:
                support.observe(store)
            except Exception as exc:
                error = exc

        with self._state:
            drive_busy = self._inflight is not None
        kinds: list[WorkKind] = []
        if support.project is not None and support.projection_consumers != ():
            kinds.append("project")
        if support.drive is not None and not drive_busy:
            kinds.append("drive")
        if not kinds:
            return error

        def project(session_id: str, consumer: str) -> None:
            assert support.project is not None
            support.project(store, session_id, consumer)

        def project_claimed(tenant_id: str, session_id: str, consumer: str) -> None:
            assert support.project is not None
            bound = store.for_tenant(tenant_id)
            support.project(bound, session_id, consumer)

        def dispatch_claimed(session_id: str, lease: Lease) -> None:
            assert support.drive is not None
            if self._stop.is_set():
                self._defer_release_unhanded(store, lease)
                return
            with self._state:
                executor = self._executor
                if executor is None or self._inflight is not None:
                    self._defer_release_unhanded(store, lease)
                    return
                try:
                    future = executor.submit(self._drive_claim, session_id, lease, support)
                except Exception:
                    self._defer_release_unhanded(store, lease)
                    raise
                self._inflight = future
                self._inflight_lease = lease
                future.add_done_callback(self._future_done)

        try:
            tick(
                store,
                dispatch_claimed=dispatch_claimed,
                project=project,
                project_claimed=project_claimed,
                page_size=support.page_size,
                failure_backoff_ms=support.failure_backoff_ms,
                claim_owner=self._owner,
                claim_ttl_ms=support.lease_ttl_ms,
                work_kinds=tuple(kinds),
                projection_consumers=support.projection_consumers,
                stop_after_claimed_drive=True,
            )
        except Exception as exc:
            error = self._combine_errors(error, exc)
        return error

    def _scan(self) -> None:
        """Run one local scan and wait for its drive claim to be finished."""

        with self._state:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("LocalWorker is already running its coordinator")
            temporary_executor = self._executor is None
            if temporary_executor:
                self._ensure_executor_locked()
        try:
            with self._open_coordinator_store() as store:
                self._configure_store(store, all_tenants=self.support.all_tenants)
                error = self._combine_errors(
                    self._reap_inflight(store),
                    self._drain_observations(store),
                    self._periodic_scan(store),
                )
                while True:
                    with self._state:
                        inflight = self._inflight
                    if inflight is None:
                        break
                    self._wake.wait(min(self.scan_interval, 0.05))
                    self._wake.clear()
                    error = self._combine_errors(error, self._reap_inflight(store))
                if error is not None:
                    with self._state:
                        self._last_error = error
                    raise ExceptionGroup("Session tick failures", [error])
        finally:
            if temporary_executor:
                with self._state:
                    executor = self._executor
                    self._executor = None
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

    def _drive_claim(self, session_id: str, lease: Lease, support: WorkerSupport) -> _DriveCompletion:
        with self._state:
            self._drive_thread_ident = current_thread().ident
        handed = False
        try:
            with support.open_store() as store:
                store = store.for_tenant(lease.tenant_id)
                self._configure_store(store, support=support)
                if self._stop.is_set():
                    return _DriveCompletion(unhanded_lease=lease)
                handed = True
                assert support.drive is not None
                support.drive(store, session_id, lease=lease)
        except Exception as exc:
            return _DriveCompletion(
                unhanded_lease=None if handed else lease,
                error=exc,
            )
        finally:
            with self._state:
                self._drive_thread_ident = None
        return _DriveCompletion()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    with self._open_coordinator_store() as store:
                        self._configure_store(store, all_tenants=self.support.all_tenants)
                        self._coordinate(store)
                    break
                except Exception as exc:
                    if self._stop.is_set():
                        if not self._fail_observations(exc):
                            self._record_error(exc)
                        break
                    self._record_error(exc)
                    self._wake.wait(self.scan_interval)
                    self._wake.clear()
        finally:
            with self._state:
                executor = self._executor
                inflight_pending = self._inflight is not None
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if inflight_pending:
                try:
                    with self._open_coordinator_store() as store:
                        self._configure_store(store, all_tenants=self.support.all_tenants)
                        self._record_error(self._reap_inflight(store))
                except Exception as exc:
                    self._record_error(exc)
                    # The executor has fully drained. If the final cleanup opener
                    # also fails, drop only the process-local completed metadata;
                    # the durable lease remains fenced by expiry and epoch.
                    with self._state:
                        self._inflight = None
                        self._inflight_lease = None
                        self._state.notify_all()
            with self._state:
                pending_observations = self._pending_observations
            if pending_observations:
                self._fail_observations(RuntimeError("LocalWorker stopped before accepted observations could run"))
            with self._state:
                self._state.notify_all()

    def _coordinate(self, store: SessionStore) -> None:
        while True:
            error = self._combine_errors(
                self._reap_inflight(store),
                self._drain_observations(store),
            )
            if self._stop.is_set():
                with self._state:
                    inflight = self._inflight
                    pending_observations = self._pending_observations
                if pending_observations:
                    self._record_error(error)
                    continue
                if inflight is None:
                    self._record_error(error)
                    break
                self._wake.wait(min(self.scan_interval, 0.05))
                self._wake.clear()
                self._record_error(error)
                continue

            scan_error = self._periodic_scan(store)
            error = self._combine_errors(error, scan_error)
            with self._state:
                self._last_error = error
            self._wake.wait(self.scan_interval)
            self._wake.clear()


__all__ = ["DriveCallback", "LocalWorker", "ObservationCallback", "WorkerSupport"]
