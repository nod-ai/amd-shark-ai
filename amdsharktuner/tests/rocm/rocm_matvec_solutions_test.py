# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from iree.compiler.dialects import iree_gpu  # type: ignore

from amdsharktuner import common, dispatch_parser
from amdsharktuner.rocm import rocm_common, rocm_solutions

from amdsharktuner.test_utils import tuner_ctx  # noqa: F401


@pytest.fixture
def gpu_target_info(tuner_ctx: common.TunerContext) -> iree_gpu.TargetInfo:
    context = tuner_ctx.mlir_ctx
    return iree_gpu.TargetInfo(
        context=context,
        arch="gfx942",
        subgroup_size_choices=[64],
        max_workgroup_sizes=[1024, 1024, 1024],
        max_thread_count_per_workgroup=1024,
        max_workgroup_memory_bytes=65536,
        workgroup_count=304,
        simds_per_workgroup=4,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        ],
    )


def _make_matvec_op_info(
    tuner_ctx: common.TunerContext,
) -> dispatch_parser.MatvecOpInfo:
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32
    return dispatch_parser.MatvecOpInfo(
        root_op=None,  # type: ignore[arg-type]
        indexing_maps=[],
        parallel_bounds=[4096],
        parallel_dim_indices=[0],
        reduction_bound=4096,
        reduction_dim_index=1,
        num_loops=2,
        largest_operand_bitwidth=16,
        lhs_type=common.ShapedType([4096, 4096], f16),
        rhs_type=common.ShapedType([4096], f16),
        res_type=common.ShapedType([4096], f32),
    )


def test_generate_matvec_solutions_yields_configurations(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    op_info = _make_matvec_op_info(tuner_ctx)
    results = list(
        rocm_solutions.generate_matvec_solutions(
            tuner_ctx=tuner_ctx,
            op_info=op_info,
            gpu_target_info=gpu_target_info,
            max_candidates=5,
        )
    )
    assert len(results) >= 1
    for config_list in results:
        assert len(config_list) == 1
        cfg = config_list[0]
        assert cfg.name == "compilation_info"
        assert isinstance(cfg.knob_assignment, rocm_common.LLVMGPUMatvecKnobs)
        info_str = str(cfg.configuration)
        assert "VectorDistribute" in info_str
        assert "partial_reduction" in info_str


def test_generate_matvec_solutions_rejects_unsupported_bitwidth(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    op_info = _make_matvec_op_info(tuner_ctx)
    op_info.largest_operand_bitwidth = 64
    results = list(
        rocm_solutions.generate_matvec_solutions(
            tuner_ctx=tuner_ctx,
            op_info=op_info,
            gpu_target_info=gpu_target_info,
            max_candidates=5,
        )
    )
    assert results == []
