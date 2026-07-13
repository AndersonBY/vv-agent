# App Server Runtime Mapping

## Runtime contracts

- `Runner.start(agent, input, run_config)` is the execution entrypoint for App Server turns.
- `RunHandle.events()` is the live runtime stream.
- `RunHandle.result()` is the terminal turn result.
- `RunHandle.cancel()`, `steer()`, `follow_up()`, `approve()`, and `state()` are the control surface.
- `RunEvent` is the runtime event contract.
- `JsonlRunEventStore` is the baseline append-only event replay store.
- `InteractiveAgentClient` remains public, but App Server should not depend on v-claw internals.

## App Server mapping

| App Server | Runtime |
| --- | --- |
| Thread | session metadata plus replay scope |
| Turn | one `RunHandle` |
| Item | `RunEvent` mapped to protocol item |
| Approval request | `ApprovalBroker` pending request routed through server request id |
| Tool-call delta | streamed `tool_call_progress` `RunEvent` metadata |
| Warning | failed run or event-stream exception |

## Constraints

- App Server must not parse raw runtime logs.
- App Server must not call `AgentRuntime` internals directly.
- App Server must not import product modules from v-claw or backend.
