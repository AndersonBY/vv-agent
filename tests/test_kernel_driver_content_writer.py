"""CoreDriver exposes content writes only through the caller's capable transaction."""

from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from vv_agent.kernel._content import ContentSessionWriter
from vv_agent.kernel._driver import CoreDriver
from vv_agent.kernel._store import ContentSessionTx, SessionTx


def _driver(*, tenant_id: str = "tenant-a") -> CoreDriver:
    driver = object.__new__(CoreDriver)
    driver.store = SimpleNamespace(tenant_id=tenant_id)
    driver.sid = "session-a"
    driver.content_reader = None
    return driver


def test_content_writer_is_bound_to_current_transaction_store_and_session() -> None:
    driver = _driver()
    tx = create_autospec(ContentSessionTx, instance=True)
    writer = create_autospec(ContentSessionWriter, instance=True)
    tx.content_writer.return_value = writer

    assert driver.content_writer(tx) is writer

    tx.content_writer.assert_called_once_with(tenant_id="tenant-a", session_id="session-a")
    assert all(value is not writer for value in vars(driver).values())


@pytest.mark.parametrize("tx", [None, create_autospec(SessionTx, instance=True)])
def test_content_writer_fails_closed_without_transaction_capability(tx) -> None:
    driver = _driver()

    with pytest.raises(ValueError, match="outer transaction with ContentSessionTx capability"):
        driver.content_writer(tx)


def test_content_writer_rejects_a_store_without_a_tenant_binding() -> None:
    driver = _driver(tenant_id="")
    tx = create_autospec(ContentSessionTx, instance=True)

    with pytest.raises(ValueError, match="non-empty tenant binding"):
        driver.content_writer(tx)

    tx.content_writer.assert_not_called()
