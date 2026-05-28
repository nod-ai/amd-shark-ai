# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ROCm-specific candidate ordering heuristics."""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from iree.compiler.dialects import iree_gpu  # type: ignore

from amdsharktuner import common

VGPR_SPILL_LIMIT = 256
VGPR_PRESSURE_TOLERANCE_RATE = 1.25
VGPR_PRESSURE_THRESHOLD = VGPR_SPILL_LIMIT / 2 * VGPR_PRESSURE_TOLERANCE_RATE
# The tuner does not know each z3 solution's mma_kind output bits, so use the
# largest likely output type as the worst-case estimate.
MAX_MMA_KIND_OUTPUT_BITS = 32


@dataclass(frozen=True)
class _VgprAttrs:
    sg_size: int
    num_mfma_c_output: float


def _indexed_product(
    solution: common.SMTKnobAssignments, prefix: str, *, skip_zero: bool = False
) -> int | None:
    values = [
        value
        for name, value in sorted(solution.items())
        if name.startswith(prefix) and name[len(prefix) :].isdigit()
        if not skip_zero or value != 0
    ]
    if not values:
        return None
    return math.prod(values)


def _get_required(solution: common.SMTKnobAssignments, name: str) -> int | None:
    value = solution.get(name)
    if value is None or value == 0:
        return None
    return value


def _get_vgpr_attrs(
    solution: common.SMTKnobAssignments,
    codegen_pipeline: iree_gpu.LoweringPipeline,
) -> _VgprAttrs | None:
    sg_size = _get_required(solution, "sg_size")
    if sg_size is None:
        logging.warning(
            "Missing required SMT symbol `sg_size`; skipping heuristic candidate ordering."
        )
        return None

    match codegen_pipeline:
        case iree_gpu.LoweringPipeline.VectorDistribute:
            wg_product = _indexed_product(solution, "wg_", skip_zero=True)
            sg_m_cnt = _get_required(solution, "sg_m_cnt")
            sg_n_cnt = _get_required(solution, "sg_n_cnt")
            if None in (wg_product, sg_m_cnt, sg_n_cnt):
                missing = []
                if wg_product is None:
                    missing.append("wg_*")
                if sg_m_cnt is None:
                    missing.append("sg_m_cnt")
                if sg_n_cnt is None:
                    missing.append("sg_n_cnt")
                logging.warning(
                    "Missing required SMT symbol(s) %s; skipping heuristic candidate ordering.",
                    ", ".join(missing),
                )
                return None
            assert wg_product is not None
            assert sg_m_cnt is not None
            assert sg_n_cnt is not None
            return _VgprAttrs(
                sg_size=sg_size,
                num_mfma_c_output=wg_product / (sg_m_cnt * sg_n_cnt),
            )
        case iree_gpu.LoweringPipeline.TileAndFuse:
            subgroup_product = _indexed_product(solution, "sg_")
            if subgroup_product is None:
                logging.warning(
                    "Missing required SMT symbol(s) `sg_*`; skipping heuristic candidate ordering."
                )
                return None
            return _VgprAttrs(
                sg_size=sg_size,
                num_mfma_c_output=float(subgroup_product),
            )
        case _:
            return None


def _get_vgpr_pressure(attrs: _VgprAttrs) -> float:
    vgpr_usage_per_thread = VGPR_SPILL_LIMIT * MAX_MMA_KIND_OUTPUT_BITS / attrs.sg_size
    return vgpr_usage_per_thread * attrs.num_mfma_c_output


def _vgpr_pressure_sort_key(attrs: _VgprAttrs) -> tuple[int, float]:
    vgpr_pressure = _get_vgpr_pressure(attrs)
    # Prefer larger VGPR pressure while below the spill threshold, then prefer
    # smaller pressure once candidates exceed the threshold.
    if vgpr_pressure <= VGPR_PRESSURE_THRESHOLD:
        return (0, -vgpr_pressure)
    return (1, vgpr_pressure)


def get_heuristic_key_fn(
    solutions: Sequence[common.SMTKnobAssignments],
    codegen_pipeline: iree_gpu.LoweringPipeline,
    dispatch_kind: common.DispatchKind,
) -> Callable[[int], tuple[int, float]] | None:
    # Experimental heuristic sorting: estimate relative VGPR pressure from SMT
    # symbols and prefer candidates that are likely to avoid spills.
    if dispatch_kind not in (common.DispatchKind.contraction, common.DispatchKind.conv):
        logging.warning(
            f"ROCm heuristic candidate ordering is not yet supported for "
            f"{dispatch_kind.name} dispatches."
        )
        return None

    if codegen_pipeline not in (
        iree_gpu.LoweringPipeline.VectorDistribute,
        iree_gpu.LoweringPipeline.TileAndFuse,
    ):
        logging.warning(
            f"No ROCm heuristic candidate ordering is defined for "
            f"{codegen_pipeline.name}."
        )
        return None

    attrs_by_index = [
        _get_vgpr_attrs(solution, codegen_pipeline) for solution in solutions
    ]
    if any(attrs is None for attrs in attrs_by_index):
        logging.warning(
            "Heuristic candidate ordering skipped because at least "
            "one solution is missing required SMT symbols."
        )
        return None

    valid_attrs = [attrs for attrs in attrs_by_index if attrs is not None]

    def heuristic_key(index: int) -> tuple[int, float]:
        return _vgpr_pressure_sort_key(valid_attrs[index])

    return heuristic_key
