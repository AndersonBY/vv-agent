# App Server runtime mapping

App Server owns transport connections, subscriptions, protocol formatting and
approval routing. SessionDriver and the kernel own execution. Product modules
stay behind AppServerHost agent/configuration providers.

| App Server | Kernel owner |
| --- | --- |
| Thread | Session record metadata and projected ThreadStatus |
| Turn | Retained kernel turn identity plus process-local RunHandle |
| Item | Typed record-derived event projected to protocol item |
| Approval | Original owner and deadline retained in the parked operation; authenticated inbox answer |
| Live delta | Volatile RunEvent observation routed to subscribed transports |
| Read/resume | Retained records and stable item cursor; execution resume drives the original turn |

SessionRunEventStore supplies durable event history. JsonlRunEventStore is an
optional sink. thread_store/thread_state hold projected values and local handles;
they own no queue, transcript or execution ledger. Host controls enter the inbox.
