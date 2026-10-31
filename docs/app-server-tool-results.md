# Reading tool results

A completed `toolCall` item contains `operationId`, `attempt` and `result`.
The result has a type, a short preview, a byte count and a reference to its
immutable completion record. The same descriptor appears in live items,
`thread/read` and `thread/events/list`. The `tool_result` diagnostic contains
this descriptor too; it no longer carries the full body or arbitrary metadata.

Use `thread/toolResult/read` to read the body:

```python
from vv_agent.app_server import ToolResultReadParams, ToolResultRef

page = client.read_tool_result(
    ToolResultReadParams(
        thread_id=thread_id,
        ref=ToolResultRef.from_dict(item["payload"]["result"]["ref"]),
        part="content",
        offset=0,
        limit=65536,
    )
)
```

Decode `data` as base64 and append the bytes. Continue at `nextOffset` until
`eof`. Pages can split a UTF-8 character; decode text after joining the bytes.
`totalBytes` and `sha256` describe the whole selected body. The response also
identifies the original `normal`, `correction` or `audit` completion, matching
the descriptor's `context` field.
Reading a result starts no tool or model work.

For a file or image, `content` describes its public resources. Read their
advertised IDs, such as `resource:0`, as `part`. A text artifact is marked
`fullText=true`; the content part remains the recorded preview. A read-file
cursor exposes its character offset and digest, without a private path.
It does not grant access to the rest of the source file.

A truncated JSON/error result can have a preview that is not valid JSON.
Read its `fullText` artifact resource to recover the original document.

| Limit | Value |
| --- | --- |
| Preview | 2,048 UTF-8 bytes |
| Inline JSON value | 8,192 canonical JSON bytes |
| Result descriptor | 16 KiB |
| Resources per result | 8 |
| Resource name and media type | 256 UTF-8 bytes each |
| Page | 1..65,536 data bytes |
| Response including JSON-RPC envelope | 96 KiB |
| Offset and byte count | Integer in 0..2^53-1 |

The reference binds the session, turn, operation, attempt, tool call,
record sequence and record digest. It always selects that exact completion,
including an older attempt. Child results also require the original admitted
child graph and permission for that child.

Resource reads rebuild host dependencies with the original admitting turn's
metadata. A later turn can select another workspace without redirecting an
old result to that workspace. Permission checks still use the current host.

## Keeping output resources

Workspace backends opt in through `OutputRetentionBackend`. The kernel calls
`retain_output(path, expected=...)` only for typed file/image outputs and
existing text artifacts. The backend returns a `ToolResultResourceRef` with
its workspace identity, the supplied owner, a private locator, size, digest,
media type and name. It returns `None` when it does not retain outputs.
No capability means `not_retained`; the kernel does not make a fallback copy.

An immutable object backend should name its existing version and pin it against
garbage collection for as long as the receipt remains readable. It can use
the version's recorded size and digest without reading or copying the object
on tool completion. `open_retained(ref)` yields byte chunks of at most
65,536 bytes. It must open that version, check workspace/owner boundaries and
reject unauthorized locators. The server checks the full digest and length
before returning the requested page.

Local, Memory and S3 backends provide private exclusive snapshots under
`.vv-agent/resources/`. Local snapshots share the private artifact root;
set `VV_AGENT_PRIVATE_ARTIFACT_ROOT` to durable host storage when results must
survive host replacement. Existing text artifacts already have immutable
references and do not need another copy. Ordinary workspace discovery and
writes cannot expose or overwrite retained objects.

The default Memory backend survives only while the host keeps that backend
instance. A durable session database does not preserve an ephemeral workspace.
Remote image URLs are not fetched or returned as download instructions.
Resource preparation failure records `unavailable`, preserving the tool's
original success or failure without running it again.

The private refs live in `ToolExecutionResult.metadata.app_server_resources`.
Only the kernel writes this reserved metadata. Product serializers should
allowlist their own fields instead of returning it to users.

## Permission and errors

Every page calls `AppServerHost.authorize`. Resource pages also call the
required `authorize_tool_result_resource(request)` hook before opening bytes.
The request includes the connection, outer thread, verified result reference,
workspace backend and retained resource. The default host allows only the
backend's own workspace. A product host can further restrict access by user,
tenant, task or file policy.

| Reason | JSON-RPC code |
| --- | --- |
| `permission_denied` | -32027 |
| `invalid_result_ref` | -32602 |
| `invalid_range` | -32602 |
| `result_not_found` | -32602 |
| `resource_not_retained` | -32602 |
| `resource_missing` | -32602 |
| `stale_reference` | -32602 |

The reason is also in `error.data.reason`. A missing object and a changed
digest are reported only after permission checks. Deleted or overwritten source
paths do not change retained resources. Historical file results without a
snapshot remain `not_retained`; the server never reads today's file as an
old output.

File reads keep only the requested page and one bounded stream chunk in
memory. Text reads still decode a complete SQL record; their page limit is
not a bound on the record decoder's allocation.
