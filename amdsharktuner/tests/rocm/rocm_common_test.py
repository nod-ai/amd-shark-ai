# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import math
import pytest

from amdsharktuner.rocm import rocm_common


def test_compute_rocprof_avg_kernel_time(caplog):
    with pytest.raises(ValueError):
        rocm_common.compute_rocprof_avg_kernel_time([])

    trace_rows = [
        {"Kernel_Name": "main_kernel", "Start_Timestamp": "0"},
        {"Kernel_Name": "main_kernel", "Start_Timestamp": "1000"},
    ]
    with pytest.raises(ValueError):
        rocm_common.compute_rocprof_avg_kernel_time(trace_rows)

    trace_rows = [
        {
            "Kernel_Name": "main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_1024x1280x1280_f16xf16xf32_buffer",
            "Start_Timestamp": "0",
            "End_Timestamp": "1000",
        },
        {
            "Kernel_Name": "main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_1024x1280x1280_f16xf16xf32",
            "Start_Timestamp": "1000",
            "End_Timestamp": "3000",
        },
    ]
    with pytest.raises(RuntimeError):
        rocm_common.compute_rocprof_avg_kernel_time(trace_rows)

    drop_row = {
        "Kernel_Name": "main_kernel",
        "Start_Timestamp": "0",
        "End_Timestamp": "1000",
    }
    cal_row = {
        "Kernel_Name": "main_kernel",
        "Start_Timestamp": "2000",
        "End_Timestamp": "3500",
    }
    cal_row_2 = {
        "Kernel_Name": "main_kernel",
        "Start_Timestamp": "4000",
        "End_Timestamp": "6000",
    }
    trace_rows = [drop_row] * 10
    with caplog.at_level(logging.WARNING):
        rocm_common.compute_rocprof_avg_kernel_time(trace_rows)

    trace_rows = [drop_row] * 10 + [cal_row] * 5 + [cal_row_2] * 5
    avg_us = rocm_common.compute_rocprof_avg_kernel_time(trace_rows)
    assert avg_us == pytest.approx(1.75)


def test_class_RocProfBenchmarkResult():
    result = rocm_common.RocProfBenchmarkResult(
        candidate_id=0,
        time=math.nan,
        device_id="0",
    )
    assert result.is_valid() == False
    result.time = math.inf
    assert result.is_valid() == False
    result.time = 1.0
    assert result.is_valid() == True
    # Test __iter__.
    candidate_id, time, device_id = result
    assert candidate_id == 0
    assert time == 1.0
    assert device_id == "0"
