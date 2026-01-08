# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import multiprocessing
from typing import Any


def pytest_sessionstart(session: Any) -> None:
    try:
        # Use "spawn" to avoid fork() warnings in pytest.
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Start method already set or multiprocessing already initialized.
        pass
