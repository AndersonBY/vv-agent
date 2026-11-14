"""Tool intents and child dependencies reuse the normal model operation ledger."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vv_agent.kernel._summary_requests import compaction_request as compaction_request
from vv_agent.kernel._summary_requests import pending_compaction_requests as pending_compaction_requests
from vv_agent.kernel._summary_requests import summary_pending as summary_pending

from .children import SummaryChildTool
from .records import Record

if TYPE_CHECKING:
    from .kernel import _Driver


def child_summary(driver: _Driver, plan: Record, binding: SummaryChildTool) -> tuple[bool, str | None]:
    from vv_agent.kernel._child_summary import child_summary as summary_steps

    from ._driver_dispatch import _DispatchHost

    return _DispatchHost(driver, binding=binding).run(summary_steps(driver, plan))
