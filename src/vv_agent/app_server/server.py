from __future__ import annotations

from vv_agent.app_server.outgoing import OutgoingRouter
from vv_agent.app_server.processor import MessageProcessor
from vv_agent.app_server.transport import StdioJsonlTransport


class AppServer:
    def __init__(
        self,
        *,
        transport: StdioJsonlTransport | None = None,
        router: OutgoingRouter | None = None,
        processor: MessageProcessor | None = None,
    ) -> None:
        self.transport = transport or StdioJsonlTransport()
        self.router = router or OutgoingRouter()
        self.processor = processor or MessageProcessor(router=self.router)
        self.router.register_transport(self.transport)

    def run_forever(self) -> None:
        for payload in self.transport.read_messages():
            self.processor.process_message(self.transport.connection_id, payload)
