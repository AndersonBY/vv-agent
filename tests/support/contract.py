"""Select local canonical vectors without changing an immutable release lock."""

import os
from pathlib import Path


def fixture_root() -> Path:
    source = os.environ.get("VV_AGENT_CONTRACT_SOURCE")
    return Path(source) / "fixtures" if source else Path(__file__).parents[1] / "fixtures/parity"
