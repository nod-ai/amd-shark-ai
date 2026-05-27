# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler.dialects import iree_gpu  # type: ignore

from amdsharktuner import common
from amdsharktuner.rocm import rocm_candidate_ordering


def test_vd_vgpr_pressure() -> None:
    solution = common.SMTKnobAssignments(
        {
            "sg_size": 64,
            "wg_0": 16,
            "wg_1": 4,
            "wg_2": 0,
            "sg_m_cnt": 4,
            "sg_n_cnt": 1,
        }
    )

    assert (
        rocm_candidate_ordering.get_vgpr_pressure(
            solution,
            iree_gpu.LoweringPipeline.VectorDistribute,
            common.DispatchKind.contraction,
        )
        == 2048.0
    )


def test_tf_vgpr_pressure() -> None:
    solution = common.SMTKnobAssignments(
        {
            "sg_size": 64,
            "sg_0": 2,
            "sg_1": 4,
            "sg_2": 1,
        }
    )

    assert (
        rocm_candidate_ordering.get_vgpr_pressure(
            solution,
            iree_gpu.LoweringPipeline.TileAndFuse,
            common.DispatchKind.conv,
        )
        == 1024.0
    )


def test_vgpr_pressure_sort_key_prefers_high_under_threshold_then_low_over() -> None:
    under_low = common.SMTKnobAssignments(
        {"sg_size": 64, "wg_0": 1, "sg_m_cnt": 1, "sg_n_cnt": 1}
    )
    under_high = common.SMTKnobAssignments(
        {"sg_size": 64, "wg_0": 1, "sg_m_cnt": 2, "sg_n_cnt": 1}
    )
    over_low = common.SMTKnobAssignments(
        {"sg_size": 64, "wg_0": 2, "sg_m_cnt": 1, "sg_n_cnt": 1}
    )
    over_high = common.SMTKnobAssignments(
        {"sg_size": 64, "wg_0": 4, "sg_m_cnt": 1, "sg_n_cnt": 1}
    )
    solutions = [over_high, under_low, over_low, under_high]

    assert sorted(
        solutions,
        key=lambda solution: rocm_candidate_ordering.vgpr_pressure_sort_key(
            solution,
            iree_gpu.LoweringPipeline.VectorDistribute,
            common.DispatchKind.contraction,
        ),
    ) == [under_low, under_high, over_low, over_high]


def test_vgpr_pressure_not_applied_to_attention() -> None:
    solution = common.SMTKnobAssignments(
        {
            "sg_size": 64,
            "wg_0": 16,
            "wg_1": 4,
            "sg_m_cnt": 4,
            "sg_n_cnt": 1,
            "qk_mma_idx": 0,
            "pv_mma_idx": 0,
        }
    )

    assert (
        rocm_candidate_ordering.get_vgpr_pressure(
            solution,
            iree_gpu.LoweringPipeline.VectorDistribute,
            common.DispatchKind.attention,
        )
        is None
    )


def test_missing_symbols_disable_vgpr_pressure() -> None:
    solution = common.SMTKnobAssignments(
        {
            "sg_size": 64,
            "sg_m_cnt": 4,
            "sg_n_cnt": 1,
        }
    )

    assert (
        rocm_candidate_ordering.get_vgpr_pressure(
            solution,
            iree_gpu.LoweringPipeline.VectorDistribute,
            common.DispatchKind.contraction,
        )
        is None
    )
    assert (
        rocm_candidate_ordering.vgpr_pressure_sort_key(
            solution,
            iree_gpu.LoweringPipeline.VectorDistribute,
            common.DispatchKind.contraction,
        )
        is None
    )


def test_get_heuristic_key_fn_prevalidates_all_solutions() -> None:
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
