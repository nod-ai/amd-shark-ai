# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from amdsharktuner import process_utils


def test_worker_context_manager_set_and_get():
    ctx = process_utils.WorkerContext(worker_id=0, device_id="hip://1")
    process_utils.WorkerContextManager.set(ctx)
    assert process_utils.WorkerContextManager.get() == ctx
    assert ctx.worker_id == 0
    assert ctx.device_id == "hip://1"


def test_worker_context_manager_initializer_pulls_context():
    ctx_manager = process_utils.WorkerContextManager(device_ids=["hip://2", "hip://5"])

    # Call initializer manually (simulating a multiprocessing worker).
    ctx_manager.initializer()
    ctx = process_utils.WorkerContextManager.get()

    assert ctx.worker_id in [0, 1]
    assert ctx.device_id in ["hip://2", "hip://5"]
