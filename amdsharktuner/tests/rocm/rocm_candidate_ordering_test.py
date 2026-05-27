# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler.dialects import iree_gpu  # type: ignore

from amdsharktuner import common
from amdsharktuner.rocm import rocm_candidate_ordering


def test_missing_symbols_disable_vgpr_pressure() -> None:
    solutions = [
        common.SMTKnobAssignments(
            {"sg_size": 64, "wg_0": 2, "sg_m_cnt": 1, "sg_n_cnt": 1}
        ),
        common.SMTKnobAssignments({"sg_size": 64, "sg_m_cnt": 1, "sg_n_cnt": 1}),
    ]

    assert (
        rocm_candidate_ordering.get_heuristic_key_fn(
            solutions,
            iree_gpu.LoweringPipeline.VectorDistribute,
            common.DispatchKind.contraction,
        )
        is None
    )


def test_get_heuristic_key_fn_sorts_with_precomputed_attrs() -> None:
    solutions = [
        common.SMTKnobAssignments(
            {"sg_size": 64, "wg_0": 4, "sg_m_cnt": 1, "sg_n_cnt": 1}
        ),
        common.SMTKnobAssignments(
            {"sg_size": 64, "wg_0": 1, "sg_m_cnt": 1, "sg_n_cnt": 1}
        ),
        common.SMTKnobAssignments(
            {"sg_size": 64, "wg_0": 2, "sg_m_cnt": 1, "sg_n_cnt": 1}
        ),
    ]

    key_fn = rocm_candidate_ordering.get_heuristic_key_fn(
        solutions,
        iree_gpu.LoweringPipeline.VectorDistribute,
        common.DispatchKind.contraction,
    )

    assert key_fn is not None
    assert sorted(range(len(solutions)), key=key_fn) == [1, 2, 0]
