from __future__ import annotations

import pytest

from vv_agent.session.surfaces import _SessionKernel


@pytest.fixture(params=["current", "kernel"])
def surface(request, monkeypatch):
    kernel = _SessionKernel() if request.param == "kernel" else None
    module = request.module
    servers = []
    if hasattr(module, "AppServer"):
        original_server = module.AppServer

        def server_factory(**kwargs):
            kwargs.setdefault("_kernel", kernel)
            server = original_server(**kwargs)
            servers.append(server)
            return server

        monkeypatch.setattr(module, "AppServer", server_factory)
    if hasattr(module, "InteractiveAgentClient"):
        original_client = module.InteractiveAgentClient
        monkeypatch.setattr(module, "InteractiveAgentClient", lambda **kwargs: original_client(**kwargs, _kernel=kernel))
    yield kernel
    if kernel:
        for server in servers:
            for handle in kernel.handles:
                handle.cancel("test teardown")
            server.router.cancel_matching_server_requests()
            server.run_adapter.join()
        kernel.close()
