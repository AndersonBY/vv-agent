from __future__ import annotations

from typing import Any

from vv_agent.app_server.host import AppServerHost, DefaultAppServerHost
from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.processor import MessageProcessor
from vv_agent.app_server.run_adapter import RunAdapter
from vv_agent.app_server.thread_state import ThreadStateManager
from vv_agent.app_server.thread_store import ThreadStore
from vv_agent.app_server.transport import AppServerTransport, StdioJsonlTransport


class AppServer:
    def __init__(
        self,
        *,
        transport: AppServerTransport | None = None,
        host: AppServerHost | None = None,
        store: ThreadStore | None = None,
        state_manager: ThreadStateManager | None = None,
        router: OutgoingRouter | None = None,
        processor: MessageProcessor | None = None,
        _kernel: Any = None,
    ) -> None:
        self.transport = transport or StdioJsonlTransport()
        self.host = host or DefaultAppServerHost()
        if _kernel is not None:
            from vv_agent.session.app_server import _KernelThreadStore

            if store is not None:
                raise ValueError("kernel threads are projections, not a second ThreadStore")
            self.store = _KernelThreadStore(_kernel)
        else:
            self.store = store or ThreadStore()
        self.state_manager = state_manager or ThreadStateManager()
        self.router = router or OutgoingRouter()
        if _kernel is not None:
            from vv_agent.session.app_server import _KernelRunAdapter

            self.run_adapter = _KernelRunAdapter(
                kernel=_kernel, host=self.host, store=self.store, state_manager=self.state_manager, router=self.router
            )
        else:
            self.run_adapter = RunAdapter(host=self.host, store=self.store, state_manager=self.state_manager, router=self.router)
        self.processor = processor or MessageProcessor(
            router=self.router,
            host=self.host,
            store=self.store,
            state_manager=self.state_manager,
            run_adapter=self.run_adapter,
        )
        self.router.register_transport(self.transport)

    def run_forever(self) -> None:
        connection_id = self.transport.connection_id
        try:
            for payload in self.transport.read_messages():
                if not isinstance(payload, dict):
                    raise ValueError("App Server transport yielded a non-object payload")
                typed_payload: dict[str, Any] = payload
                self.processor.process_message(connection_id, typed_payload)
                if not self.router.is_registered(connection_id):
                    return
        finally:
            self.processor.disconnect_connection(connection_id)
