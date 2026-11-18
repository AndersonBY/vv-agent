"""The W9.6 alias preserves values without shadowing stdlib types at startup."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FLAGS = [[], ["-S"], ["-S", "-X", "frozen_modules=off"]]
_MODES = ["normal", "no-site", "unfrozen"]


def _python(script, flags, *, cwd):
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "PYTHONSTARTUP"}}
    env["PYTHONPATH"] = str(_ROOT / "src")
    result = subprocess.run(
        [sys.executable, *flags, "-B", "-c", script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("flags", _FLAGS, ids=_MODES)
def test_stdlib_types_from_real_package_cwd(tmp_path, flags):
    baseline = json.loads(
        _python(
            "import types, json; print(json.dumps({'names': sorted(vars(types)), 'file': types.__file__, "
            "'cached': types.__cached__}))",
            ["-I", "-S"],
            cwd=tmp_path,
        )
    )
    script = textwrap.dedent("""
        import sys

        if sys.flags.no_site:
            assert "types" not in sys.modules, "no-site regression must exercise a fresh types import"

        class BootstrapOnly:
            def find_spec(self, fullname, path=None, target=None):
                assert fullname not in {"vv_agent", "sysconfig", "importlib.util"}, fullname
                assert not fullname.startswith("vv_agent."), fullname

        fence = BootstrapOnly()
        sys.meta_path.insert(0, fence)
        import types
        sys.meta_path.remove(fence)

        import functools, typing, enum, dataclasses, importlib

        assert types.GenericAlias is type(list[int])
        assert types.MappingProxyType is type(type.__dict__)
        assert types.ModuleType is type(sys)
        assert not any(name == "vv_agent" or name.startswith("vv_agent.") for name in sys.modules)
        assert types.__spec__.origin == types.__file__ == types.__loader__.get_filename("types")
        assert types.__package__ == ""
        assert sorted(vars(types)) == BASELINE["names"]
        assert types.__file__ == BASELINE["file"]
        assert types.__cached__ == BASELINE["cached"]
        assert importlib.reload(types) is types is sys.modules["types"]
        assert sorted(vars(types)) == BASELINE["names"]
        assert types.__spec__.origin == types.__file__ == BASELINE["file"]
    """).replace("BASELINE", repr(baseline))
    _python(script, flags, cwd=_ROOT / "src/vv_agent")


@pytest.mark.parametrize("flags", _FLAGS, ids=_MODES)
@pytest.mark.parametrize("order", ["stdlib-first", "package-first"])
def test_package_aliases_preserve_all_existing_bindings(flags, order):
    first = "import types\n" if order == "stdlib-first" else "import vv_agent.types\n"
    script = first + textwrap.dedent("""
        import sys
        import types
        from vv_agent import core_types as current
        from vv_agent import types as legacy
        from vv_agent import Message, AgentTask

        bindings = {name: value for name, value in vars(current).items() if not name.startswith("__")}
        assert len(bindings) == 47
        assert bindings == {name: value for name, value in vars(legacy).items() if not name.startswith("__")}
        for name, value in bindings.items():
            assert getattr(legacy, name) is value, name
            assert not hasattr(types, name), name
        assert Message is current.Message is legacy.Message
        assert AgentTask is current.AgentTask is legacy.AgentTask
        assert types.GenericAlias is type(list[int])
        assert types.ModuleType is type(sys)
        assert sys.modules["types"] is types
    """)
    _python(script, flags, cwd=_ROOT / "src/vv_agent")


@pytest.mark.parametrize("package_cwd", [False, True], ids=["stdlib-control", "package-cwd"])
def test_concurrent_types_import_waits_for_stdlib_initialization(tmp_path, package_cwd):
    script = textwrap.dedent("""
        import sys, os, _thread

        assert "types" not in sys.modules
        assert not sys.flags.safe_path
        stdlib = os.path.join(os.path.dirname(os.__file__), "types.py")
        entered = _thread.allocate_lock(); entered.acquire()
        release = _thread.allocate_lock(); release.acquire()
        attempted = _thread.allocate_lock(); attempted.acquire()
        first_done = _thread.allocate_lock(); first_done.acquire()
        second_done = _thread.allocate_lock(); second_done.acquire()
        observed, errors, specs = [], [], []

        def pause_at_stdlib_entry(frame, event, arg):
            if event == "call" and frame.f_code.co_name == "<module>" and frame.f_globals.get("__name__") == "types":
                specs.append(frame.f_globals["__spec__"])
                if frame.f_code.co_filename == stdlib:
                    sys.settrace(None)
                    entered.release()
                    assert release.acquire(timeout=5)
            return pause_at_stdlib_entry

        def first():
            try:
                sys.settrace(pause_at_stdlib_entry)
                import types
                assert hasattr(types, "GenericAlias")
            except BaseException as error:
                errors.append(("first", type(error).__name__, str(error)))
            finally:
                sys.settrace(None)
                first_done.release()

        def second():
            try:
                attempted.release()
                import types
                observed.append(hasattr(types, "GenericAlias"))
            except BaseException as error:
                errors.append(("second", type(error).__name__, str(error)))
            finally:
                second_done.release()

        _thread.start_new_thread(first, ())
        assert entered.acquire(timeout=5), errors
        _thread.start_new_thread(second, ())
        assert attempted.acquire(timeout=5), errors
        returned_early = second_done.acquire(timeout=0.25)
        release.release()
        assert first_done.acquire(timeout=5), errors
        if not returned_early:
            assert second_done.acquire(timeout=5), errors
        assert not errors, errors
        assert not returned_early, "concurrent import returned a partially initialized types module"
        assert observed == [True], observed
        assert specs and all(spec is specs[0] for spec in specs), "importlib's original spec was replaced"
        assert not any(name == "vv_agent" or name.startswith("vv_agent.") for name in sys.modules)
    """)
    _python(script, ["-S"], cwd=_ROOT / "src/vv_agent" if package_cwd else tmp_path)
