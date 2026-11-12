import os

import pytest

from .report import database


@pytest.fixture(params=["sqlite", "postgres"])
def sim_database(request):
    if request.param == "postgres" and not os.environ.get("VV_AGENT_TEST_POSTGRES_DSN"):
        pytest.skip("set VV_AGENT_TEST_POSTGRES_DSN to a disposable PostgreSQL admin database")
    with database(request.param) as db:
        yield db
