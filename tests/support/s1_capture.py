"""Authenticated content access for deterministic S1 producer fixtures."""

import base64
from typing import Any

from vv_agent.kernel._content import ContentRef
from vv_agent.kernel._content_sql import _content_mac


class CapturedContent:
    def __init__(self, catalog: dict[str, Any]) -> None:
        self.entries = catalog["contents"]
        self.keys = {key["key_id"]: base64.b64decode(key["key_base64"], validate=True) for key in catalog["keys"]}

    def read(self, tenant_id: str, session_id: str, owner: str, slot: str, ref: ContentRef) -> bytes:
        entry = self.entries[ref.content_id]
        assert (entry["tenant_id"], entry["session_id"], entry["key_id"]) == (tenant_id, session_id, ref.key_id), (owner, slot)
        body = base64.b64decode(entry["bytes_base64"], validate=True)
        assert _content_mac(self.keys[ref.key_id], tenant_id, session_id, body) == ref.mac, (owner, slot)
        return body
