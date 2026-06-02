from __future__ import annotations

from typing import Any, cast

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
    ) -> None:
        self.transport = transport or StdioJsonlTransport()
        self.host = host or DefaultAppServerHost()
        self.store = store or ThreadStore()
        self.state_manager = state_manager or ThreadStateManager()
        self.router = router or OutgoingRouter()
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
        read_messages = cast(Any, self.transport).read_messages
        for payload in read_messages():
            if not isinstance(payload, dict):
                raise ValueError("App Server transport yielded a non-object payload")
            typed_payload: dict[str, Any] = payload
            self.processor.process_message(self.transport.connection_id, typed_payload)
