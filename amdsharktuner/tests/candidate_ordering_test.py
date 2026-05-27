# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

from amdsharktuner import candidate_ordering


def test_math_expression() -> None:
    assert candidate_ordering.is_pow2(1) == True
    assert candidate_ordering.is_pow2(5) == False
    assert candidate_ordering.is_pow2(32) == True
    assert candidate_ordering.is_pow2(6) == False

    assert candidate_ordering.is_mult_simd_num(6, 4) == False
    assert candidate_ordering.is_mult_simd_num(8, 4) == True

    ai = candidate_ordering.arith_intensity(2, 3, 4)
    expected = (2 * 2 * 3 * 4) / (2 * (2 * 3 + 3 * 4 + 2 * 4))
    assert math.isclose(ai, expected, rel_tol=1e-9)

    q_ie = candidate_ordering.quantization_inefficiency(2048, 256, 1024, 32, 32)
    assert q_ie == 0
    q_ie = candidate_ordering.quantization_inefficiency(10, 4, 10, 4, 4)
    assert q_ie == 0.21875

    assert candidate_ordering.size_ratio(256, 32) == 0.125
    assert candidate_ordering.size_ratio(32, 256) == 0.125
    assert candidate_ordering.size_ratio(32, 1024) == 0.03125
    assert candidate_ordering.size_ratio(256, 256) == 1


def test_prepare_record_csv_data_keeps_knobless_candidates() -> None:
    records = [
        candidate_ordering.TuningRecord(gen_id=0, candidate_id=0),
        candidate_ordering.TuningRecord(
            gen_id=1,
            candidate_id=1,
            compile_status=True,
            benchmark_status=True,
            benchmark_time_us=876.0,
        ),
    ]

    headers, rows = candidate_ordering.prepare_record_csv_data(records)

    assert "candidate_id" in headers
    assert rows == [
        {
            "gen_id": 1,
            "candidate_id": 1,
            "knob": None,
            "to_compile": False,
            "compile_status": True,
            "to_benchmark": False,
            "benchmark_device_id": None,
            "benchmark_queue_position": None,
            "benchmark_status": True,
            "baseline_benchmark_time_us": None,
            "benchmark_time_us": 876.0,
            "benchmark_speedup": None,
            "benchmark_rank_order": None,
        }
    ]
