from __future__ import annotations
from typing import Any

from amdsharktuner.test_utils import configure_pytest_multiprocessing


def pytest_sessionstart(session: Any) -> None:
    configure_pytest_multiprocessing()
