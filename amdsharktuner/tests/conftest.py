# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import Any

from amdsharktuner.test_utils import configure_pytest_multiprocessing


def pytest_sessionstart(session: Any) -> None:
    configure_pytest_multiprocessing()
