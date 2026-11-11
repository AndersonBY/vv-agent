"""Observable async admission/hook bindings reject before any execution work."""

from functools import partial, wraps

import pytest

from vv_agent.run_handle import RunHandle
from vv_agent.session import kernel
from vv_agent.session.callbacks import CallbackContractError, synchronous, validate
from vv_agent.session.children import ChildSession, SummaryChildTool
from vv_agent.session.records import SessionSpec
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse

from .test_compaction import SUMMARY
from .test_recovery_matrix import MODEL_USAGE, start
from .test_summary_requests import call, configured, summary_plans

ROLES = ["admission", "summary", "hook"]
FORMS = ["direct", "bound", "instance", "partial", "nested_partial"]


def binding(role, form, asynchronous, calls):
    def result(tx):
        calls.append(tx)
        return None if role == "hook" else ChildSession(SessionSpec("async-control-child", "test", "/tmp"), "go")

    if role == "admission":

        def direct(plan, *, tx):
            return result(tx)

        async def async_direct(plan, *, tx):
            return result(tx)

        class Instance:
            def __call__(self, plan, *, tx):
                return result(tx)

        class AsyncInstance:
            async def __call__(self, plan, *, tx):
                return result(tx)
    else:

        def direct(first, second, *, tx):
            return result(tx)

        async def async_direct(first, second, *, tx):
            return result(tx)

        class Instance:
            def __call__(self, first, second, *, tx):
                return result(tx)

        class AsyncInstance:
            async def __call__(self, first, second, *, tx):
                return result(tx)

    instance = AsyncInstance() if asynchronous else Instance()
    if form == "direct":
        return async_direct if asynchronous else direct
    if form == "bound":
        return instance.__call__
    if form == "instance":
        return instance
    if form == "partial":
        return partial(instance)
    return partial(partial(instance))


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_observable_callable_forms(role, form, asynchronous):
    calls = []
    callback = binding(role, form, asynchronous, calls)
    arity = 1 if role == "admission" else 2
    if asynchronous:
        with pytest.raises(CallbackContractError):
            validate(callback, arity)
        assert calls == []
    else:
        assert validate(callback, arity) is callback
        tx = object()
        synchronous(callback(*([None] * arity), tx=tx))
        assert calls == [tx]


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("form", FORMS)
def test_invalid_fixed_runtime_has_zero_work(store, database, monkeypatch, role, form):
    start(store)
    calls = []
    stages = []
    original_acquire = type(store).acquire
    initialize = kernel._Driver.__init__

    def acquire(*args, **kwargs):
        stages.append("acquire")
        return original_acquire(*args, **kwargs)

    def loaded(*args, **kwargs):
        stages.append("Driver")
        return initialize(*args, **kwargs)

    def model(request):
        stages.append("model")
        return LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE})

    def predicate(plan):
        stages.append("requires_summary")
        return True

    rt = configured(database, [call("child_task"), model], child=True)
    callback = binding(role, form, True, calls)
    if role == "hook":
        rt.hook = callback
    else:
        rt.children["child_task"] = SummaryChildTool(predicate, callback) if role == "summary" else callback
    monkeypatch.setattr(type(store), "acquire", acquire)
    monkeypatch.setattr(kernel._Driver, "__init__", loaded)
    with pytest.raises(CallbackContractError):
        kernel.drive(store, "s", runtime=rt)
    if role == "hook":
        with pytest.raises(CallbackContractError):
            RunHandle(SessionDriver(store=store), "s", "s/turn/initial", rt, None)
    assert stages == calls == []
    assert len(rt.llm.steps) == 2
    assert summary_plans(store) == []
    assert len(store.read_state("s")[1]) == 1


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("form", ["partial", "nested_partial"])
def test_sync_partial_roles_execute_real_callbacks(store, database, role, form):
    start(store)
    calls = []
    rt = configured(database, [call("child_task"), LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE})], child=True)
    callback = binding(role, form, False, calls)
    if role == "hook":
        rt.hook = callback
    else:
        rt.children["child_task"] = SummaryChildTool(lambda plan: True, callback) if role == "summary" else callback
    kernel.drive(store, "s", runtime=rt)
    assert calls
    if role == "hook":
        assert any(tx is not None for tx in calls) and any(tx is None for tx in calls)
    else:
        assert len(calls) == 1
        assert "async-control-child" in store.list_sessions()


@pytest.mark.parametrize("role", ROLES)
def test_wrapped_metadata_does_not_guarantee_sync_return(role):
    calls = []
    original = binding(role, "direct", True, calls)
    if role == "admission":

        @wraps(original)
        def wrapper(plan, *, tx):
            return original(plan, tx=tx)
    else:

        @wraps(original)
        def wrapper(first, second, *, tx):
            return original(first, second, tx=tx)

    arity = 1 if role == "admission" else 2
    assert validate(wrapper, arity) is wrapper
    value = wrapper(*([None] * arity), tx=None)
    with pytest.raises(CallbackContractError):
        synchronous(value)
    assert value.cr_frame is None and calls == []


@pytest.mark.parametrize("role", ROLES)
def test_replaced_async_partial_rejects_at_next_use(store, database, role):
    start(store)
    calls = []
    predicates = []
    rt = configured(database, [call("child_task")], child=True)
    response = rt.llm.steps[0]

    def replace_binding(request):
        callback = binding(role, "nested_partial", True, calls)
        if role == "hook":
            rt.hook = callback
        else:
            rt.children["child_task"] = (
                SummaryChildTool(lambda plan: predicates.append(plan), callback) if role == "summary" else callback
            )
        return response

    rt.llm.steps[0] = replace_binding
    with pytest.raises(CallbackContractError):
        kernel.drive(store, "s", runtime=rt)
    assert calls == predicates == []
    assert not rt.llm.steps and summary_plans(store) == []
    assert store.list_sessions() == ("s",)


@pytest.mark.parametrize("role", ROLES)
def test_partial_still_validates_bound_tx_and_explicit_opaque_wrapper(role):
    callback = binding(role, "direct", False, [])
    arity = 1 if role == "admission" else 2
    with pytest.raises(CallbackContractError):
        validate(partial(callback, tx=None), arity)

    class Opaque:
        @property
        def __signature__(self):
            raise ValueError("opaque")

        def __call__(self, *args, **kwargs):
            return callback(*args, **kwargs)

    opaque = Opaque()
    with pytest.raises(CallbackContractError):
        validate(opaque, arity)
    if role == "admission":

        def wrapper(plan, *, tx):
            return opaque(plan, tx=tx)
    else:

        def wrapper(first, second, *, tx):
            return opaque(first, second, tx=tx)

    assert validate(wrapper, arity) is wrapper
