from __future__ import annotations

import pytest

from vv_agent.session.surfaces import SessionDriver


@pytest.fixture
def surface(request, monkeypatch):
    driver = SessionDriver()
    module = request.module
    servers = []
    clients = []
    if hasattr(module, "AppServer"):
        original_server = module.AppServer

        def server_factory(**kwargs):
            kwargs.setdefault("store", driver.store)
            server = original_server(**kwargs)
            servers.append(server)
            return server

        monkeypatch.setattr(module, "AppServer", server_factory)
    if hasattr(module, "InteractiveAgentClient"):
        original_client = module.InteractiveAgentClient

        def client_factory(**kwargs):
            kwargs["options"].session_store = driver.store
            client = original_client(**kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(module, "InteractiveAgentClient", client_factory)
    yield driver
    for server in servers:
        for handle in server.kernel.handles:
            handle.cancel("test teardown")
        server.router.cancel_matching_server_requests()
        server.run_adapter.join()
        server.kernel.close()
    for client in clients:
        for handle in client.driver.handles:
            handle.cancel("test teardown")
        client.driver.close()
    driver.close()


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    """Default workspace projections belong to each disposable test directory."""
    monkeypatch.chdir(tmp_path)
