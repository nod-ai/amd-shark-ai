# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from amdsharktuner import candidate_ordering
from amdsharktuner import common
from iree.compiler.dialects import iree_gpu  # type: ignore


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
            "solution": None,
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


def test_reorder_solutions_heuristic_uses_pipeline_ordering() -> None:
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

    assert candidate_ordering.reorder_solutions(
        solutions,
        candidate_ordering.CandidateOrderKind.heuristic,
        iree_gpu.LoweringPipeline.VectorDistribute,
        common.DispatchKind.contraction,
    ) == [1, 2, 0]


def test_reorder_solutions_heuristic_ignores_attention() -> None:
    solutions = [
        common.SMTKnobAssignments(
            {"sg_size": 64, "wg_0": 4, "sg_m_cnt": 1, "sg_n_cnt": 1}
        ),
        common.SMTKnobAssignments(
            {"sg_size": 64, "wg_0": 1, "sg_m_cnt": 1, "sg_n_cnt": 1}
        ),
    ]

    assert candidate_ordering.reorder_solutions(
        solutions,
        candidate_ordering.CandidateOrderKind.heuristic,
        iree_gpu.LoweringPipeline.VectorDistribute,
        common.DispatchKind.attention,
    ) == [0, 1]


def test_reorder_solutions_heuristic_skips_when_symbols_missing() -> None:
    solutions = [
        common.SMTKnobAssignments(
            {"sg_size": 64, "wg_0": 4, "sg_m_cnt": 1, "sg_n_cnt": 1}
        ),
        common.SMTKnobAssignments({"sg_size": 64, "sg_m_cnt": 1, "sg_n_cnt": 1}),
    ]

    assert candidate_ordering.reorder_solutions(
        solutions,
        candidate_ordering.CandidateOrderKind.heuristic,
        iree_gpu.LoweringPipeline.VectorDistribute,
        common.DispatchKind.contraction,
    ) == [0, 1]


def test_build_tuning_records_from_order() -> None:
    solutions = [
        common.SMTKnobAssignments({"sg_size": 64}),
        common.SMTKnobAssignments({"sg_size": 32}),
        common.SMTKnobAssignments({"sg_size": 16}),
    ]

    records = candidate_ordering.build_tuning_records_from_order(solutions, [2, 0])

    assert [(record.gen_id, record.candidate_id) for record in records] == [
        (0, 0),
        (3, 1),
        (1, 2),
    ]


def test_prepare_record_csv_data_flattens_solution() -> None:
    records = [
        candidate_ordering.TuningRecord(gen_id=0, candidate_id=0),
        candidate_ordering.TuningRecord(
            gen_id=1,
            candidate_id=1,
            solution=common.SMTKnobAssignments({"sg_size": 64, "m_tile": 128}),
        ),
    ]

    headers, rows = candidate_ordering.prepare_record_csv_data(records)

    assert "solution_sg_size" in headers
    assert rows[0]["solution_sg_size"] == 64
    assert rows[0]["solution_m_tile"] == 128
