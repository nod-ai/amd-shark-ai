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
