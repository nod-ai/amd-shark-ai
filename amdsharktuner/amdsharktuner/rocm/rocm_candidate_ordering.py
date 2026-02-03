# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ROCm-specific candidate ordering heuristics."""

from typing import Callable

from iree.compiler.dialects import iree_gpu  # type: ignore

from amdsharktuner import candidate_ordering, common
from . import rocm_common


def llvm_gpu_contraction_sort_key(
    knob: rocm_common.LLVMGPUContractionKnobs,
    target_info: iree_gpu.TargetInfo,
) -> tuple:
    """General heuristic reordering function for all architectures and pipelines."""
    return (
        not candidate_ordering.is_pow2(knob.tile_k),
        not candidate_ordering.is_mult_simd_num(
            knob.subgroup_m_cnt * knob.subgroup_n_cnt, target_info.simds_per_workgroup
        ),
        not candidate_ordering.arith_intensity(
            knob.intrinsic_mn, knob.intrinsic_mn, knob.intrinsic_k
        ),
        candidate_ordering.quantization_inefficiency(
            knob.M, knob.tile_m, knob.N, knob.tile_n, target_info.workgroup_count
        ),
        not candidate_ordering.size_ratio(knob.tile_m, knob.tile_n),
    )


# ROCm-specific sort key registry.
ROCM_SORT_KEY_MAP: dict[type[common.KnobAssignment | None], Callable | None] = {
    rocm_common.LLVMGPUContractionKnobs: llvm_gpu_contraction_sort_key,
    type(None): None,
    # TODO: Add key() for conv, attention, and other dispatch kinds.
}
