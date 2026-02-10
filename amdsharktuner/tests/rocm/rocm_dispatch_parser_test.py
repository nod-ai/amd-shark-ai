# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from amdsharktuner import common
from amdsharktuner.rocm import rocm_common

from amdsharktuner.test_utils import tuner_ctx


def test_get_mmt_tile_sizes(tuner_ctx: common.TunerContext) -> None:
    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[128, 320, 0],
        reduction=[0, 0, 32],
        subgroup_basis=[[1, 4, 1], [0, 1, 2]],
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = rocm_common.get_translation_info_config(pipeline_options, 0)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [], 0, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    lowering_config = compilation_info.lowering_config
    assert lowering_config.workgroup_tile_sizes == [128, 320, 0]
    assert lowering_config.reduction_tile_sizes == [0, 0, 32]


def test_get_conv_tile_sizes(tuner_ctx: common.TunerContext) -> None:
    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[1, 1, 464, 320, 1, 1, 0],
        reduction=[0, 0, 0, 0, 0, 0, 16],
        subgroup_basis=[[1, 1, 1, 1, 1, 1, 4], [0, 1, 2, 3, 4, 5, 6]],
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = rocm_common.get_translation_info_config(pipeline_options, 1)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [256, 1, 1], 64, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    assert compilation_info.lowering_config.workgroup_tile_sizes == [
        1,
        1,
        464,
        320,
        1,
        1,
        0,
    ]
    assert compilation_info.lowering_config.reduction_tile_sizes == [
        0,
        0,
        0,
        0,
        0,
        0,
        16,
    ]
