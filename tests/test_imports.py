import subprocess
import sys
import textwrap

import pytest


def run_script(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_storage_and_app_server_imports_do_not_load_model_sdks():
    run_script("""
        import importlib.abc
        import sys

        forbidden = {
            "vv_llm", "openai", "anthropic", "tiktoken",
            "tokenizers", "deepseek_tokenizer", "qwen_tokenizer", "psycopg",
        }
        class NoModelSDKs(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                assert fullname.split(".")[0] not in forbidden, fullname
        finder = NoModelSDKs()
        sys.meta_path.insert(0, finder)

        import vv_agent
        from vv_agent import Agent, AppServer, Runner
        from vv_agent.session.records import InboxItem, SessionSpec
        from vv_agent.session.sqlite import SQLiteStore
        from vv_agent.session.surfaces import SessionDriver

        assert "VvLlmClient" in dir(vv_agent)
        assert {"VvLlmClient", "EndpointTarget"} <= set(dir(vv_agent.llm))
        with SQLiteStore.standalone(":memory:") as store:
            store.install_schema()
            with store.atomic() as tx:
                tx.create(SessionSpec("thread", "test", "/tmp"), consumers=("events",))
                tx.push("thread", InboxItem("selection", "model_selection", {"selection": None}))
            assert len(store.peek_inbox("thread")) == 1
        driver = SessionDriver()
        try:
            assert driver.store is not None
        finally:
            driver.close()
        assert not forbidden.intersection(sys.modules)
    """)


@pytest.mark.parametrize("first_import", ["vv_agent", "vv_agent.llm", "vv_agent.llm.vv_llm_client"])
def test_lazy_model_exports_keep_original_objects(first_import):
    run_script(f"""
        import importlib
        from typing import get_type_hints

        importlib.import_module({first_import!r})
        import vv_agent
        import vv_agent.llm
        from vv_agent import *
        from vv_agent.llm import *
        from vv_agent.llm import vv_llm_client
        from vv_llm.settings import Settings

        assert vv_agent.VvLlmClient is vv_agent.llm.VvLlmClient is vv_llm_client.VvLlmClient
        assert vv_agent.llm.EndpointTarget is vv_llm_client.EndpointTarget
        assert vars(vv_agent)["VvLlmClient"] is vv_llm_client.VvLlmClient
        assert vars(vv_agent.llm)["EndpointTarget"] is vv_llm_client.EndpointTarget
        assert get_type_hints(VvLlmClient)["settings"] == Settings | None
        assert all(name in globals() for name in vv_agent.__all__)
        assert all(name in globals() for name in vv_agent.llm.__all__)
        for module in (vv_agent, vv_agent.llm):
            try:
                getattr(module, "missing_export")
            except AttributeError:
                pass
            else:
                raise AssertionError("unknown exports must raise AttributeError")
    """)
