# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from types import SimpleNamespace

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import func, iree_codegen, iree_gpu, linalg  # type: ignore

from amdsharktuner import common, dispatch_parser
from amdsharktuner.rocm import rocm_common
from amdsharktuner.test_utils import tuner_ctx


def test_get_compatible_mma_intrinsics(tuner_ctx: common.TunerContext) -> None:
    all_intrinsics = [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
        iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
        iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
    ]

    lhs = common.ShapedType([2048, 1280], tuner_ctx.type.f16)
    rhs = common.ShapedType([1280, 1280], tuner_ctx.type.f16)
    res = common.ShapedType([2048, 1280], tuner_ctx.type.f32)
    assert rocm_common.get_compatible_mma_intrinsics(lhs, rhs, res, all_intrinsics) == [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
    ]

    lhs = common.ShapedType([2048, 1280], tuner_ctx.type.i8)
    rhs = common.ShapedType([1280, 1280], tuner_ctx.type.i8)
    res = common.ShapedType([2048, 1280], tuner_ctx.type.i32)
    assert rocm_common.get_compatible_mma_intrinsics(lhs, rhs, res, all_intrinsics) == [
        iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
        iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
    ]

    lhs = common.ShapedType([64, 968, 640], tuner_ctx.type.f32)
    rhs = common.ShapedType([64, 640, 320], tuner_ctx.type.f32)
    res = common.ShapedType([64, 968, 320], tuner_ctx.type.f32)
    assert (
        rocm_common.get_compatible_mma_intrinsics(lhs, rhs, res, all_intrinsics) == []
    )

    intrinsics_with_virtual = [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_32x32x16_F8E4M3FNUZ,
    ]

    lhs = common.ShapedType([32, 16], tuner_ctx.type.f8E4M3FNUZ)
    rhs = common.ShapedType([16, 32], tuner_ctx.type.f8E4M3FNUZ)
    res = common.ShapedType([32, 32], tuner_ctx.type.f32)
    assert (
        rocm_common.get_compatible_mma_intrinsics(
            lhs, rhs, res, intrinsics_with_virtual
        )
        == []
    )

    assert rocm_common.get_compatible_mma_intrinsics(
        lhs, rhs, res, intrinsics_with_virtual, allow_virtual_mma=True
    ) == [iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_32x32x16_F8E4M3FNUZ]


def test_get_translation_info_config(tuner_ctx: common.TunerContext) -> None:
    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        mma_kind=mma_attr,
        workgroup=[4, 8, 0],
        reduction=[0, 0, 16],
        subgroup_basis=[[1, 1, 1], [0, 1, 2]],
    )
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = rocm_common.get_translation_info_config(pipeline_options, 2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [16, 16, 1], 32, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    config1_str: str = str(
        compilation_info.translation_info.configuration[common.LLVM_FUNC_ATTRS_KEY]
    )
    assert config1_str == '{"amdgpu-waves-per-eu" = "2"}'

    pipeline_options = iree_gpu.PipelineOptionsAttr.get(prefetch_num_stages=2)
    config_dict = rocm_common.get_translation_info_config(pipeline_options, 4)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [16, 16, 1], 32, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    config2_str: str = str(compilation_info.translation_info.configuration)
    assert (
        config2_str
        == '{gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}}'
    )


def test_get_padding_conv_sizes(tuner_ctx: common.TunerContext) -> None:
    # Note: Using SimpleNamespace to create lightweight mock objects for conv_dims.
    # The actual linalg.ConvolutionDimensions is a C++-backed type from IREE's
    # Python bindings, so we mock it with SimpleNamespace for testing convenience.

    # Spatial dimension last (NCHW layout).
    conv_to_igemm_info = common.ConvToIgemmInfo(
        is_batch_dim_last=False,
        is_spatial_dim_last=True,
        conv_dims=SimpleNamespace(batch=[0], input_channel=[1]),
        conv_to_igemm_dim_map={0: 0, 1: 1, 2: 2},
        input_channel_dim_to_size={1: 64},
    )
    result = rocm_common.get_padding_conv_sizes(
        bounds=[128, 64, 56],
        padding_sizes=[256, 128, 64],
        igemm_loop_iterators=['"parallel"', '"parallel"', '"parallel"'],
        conv_to_igemm_info=conv_to_igemm_info,
    )
    assert result is None

    # Batch dimension last (CHWN layout).
    conv_to_igemm_info = common.ConvToIgemmInfo(
        is_batch_dim_last=True,
        is_spatial_dim_last=False,
        conv_dims=SimpleNamespace(batch=[0, 3], input_channel=[1]),
        conv_to_igemm_dim_map={0: 0, 1: 1, 2: 2, 3: 3},
        input_channel_dim_to_size={1: 64},
    )
    result = rocm_common.get_padding_conv_sizes(
        bounds=[128, 64, 56, 32],
        padding_sizes=[256, 128, 64, 64],
        igemm_loop_iterators=['"parallel"', '"parallel"', '"parallel"', '"parallel"'],
        conv_to_igemm_info=conv_to_igemm_info,
    )
    assert result == [0, 0, 0, 64]

    # Batch dimension last with bounds divisible by padding.
    conv_to_igemm_info = common.ConvToIgemmInfo(
        is_batch_dim_last=True,
        is_spatial_dim_last=False,
        conv_dims=SimpleNamespace(batch=[0, 3], input_channel=[1]),
        conv_to_igemm_dim_map={0: 0, 1: 1, 2: 2, 3: 3},
        input_channel_dim_to_size={1: 64},
    )
    result = rocm_common.get_padding_conv_sizes(
        bounds=[128, 64, 56, 64],
        padding_sizes=[256, 128, 64, 64],
        igemm_loop_iterators=['"parallel"', '"parallel"', '"parallel"', '"parallel"'],
        conv_to_igemm_info=conv_to_igemm_info,
    )
    assert result is None

    # Normal convolution with parallel and reduction dimensions.
    conv_to_igemm_info = common.ConvToIgemmInfo(
        is_batch_dim_last=False,
        is_spatial_dim_last=False,
        conv_dims=SimpleNamespace(batch=[0], input_channel=[3]),
        conv_to_igemm_dim_map={0: 0, 1: 1, 2: 2, 3: 3},
        input_channel_dim_to_size={3: 64},
    )
    result = rocm_common.get_padding_conv_sizes(
        bounds=[128, 56, 56, 64],
        padding_sizes=[256, 64, 64, 128],
        igemm_loop_iterators=['"parallel"', '"parallel"', '"parallel"', '"reduction"'],
        conv_to_igemm_info=conv_to_igemm_info,
    )
    assert result == [256, 64, 64, 128]

    # Reduction dimension with bounds divisible by padding.
    conv_to_igemm_info = common.ConvToIgemmInfo(
        is_batch_dim_last=False,
        is_spatial_dim_last=False,
        conv_dims=SimpleNamespace(batch=[0], input_channel=[3]),
        conv_to_igemm_dim_map={0: 0, 1: 1, 2: 2, 3: 3},
        input_channel_dim_to_size={3: 128},
    )
    result = rocm_common.get_padding_conv_sizes(
        bounds=[128, 56, 56, 128],
        padding_sizes=[256, 64, 64, 128],
        igemm_loop_iterators=['"parallel"', '"parallel"', '"parallel"', '"reduction"'],
        conv_to_igemm_info=conv_to_igemm_info,
    )
    assert result == [256, 64, 64, 0]

    # Input channel size is small compared to padding size.
    conv_to_igemm_info = common.ConvToIgemmInfo(
        is_batch_dim_last=False,
        is_spatial_dim_last=False,
        conv_dims=SimpleNamespace(batch=[0], input_channel=[3]),
        conv_to_igemm_dim_map={0: 0, 1: 1, 2: 2, 3: 3},
        input_channel_dim_to_size={3: 32},
    )
    result = rocm_common.get_padding_conv_sizes(
        bounds=[128, 56, 56, 32],
        padding_sizes=[256, 64, 64, 128],
        igemm_loop_iterators=['"parallel"', '"parallel"', '"parallel"', '"reduction"'],
        conv_to_igemm_info=conv_to_igemm_info,
    )
    assert result == [256, 64, 64, 0]

    # Multiple padded parallel dims mapping to same IGEMM dim.
    conv_to_igemm_info = common.ConvToIgemmInfo(
        is_batch_dim_last=False,
        is_spatial_dim_last=False,
        conv_dims=SimpleNamespace(batch=[0], input_channel=[3]),
        conv_to_igemm_dim_map={0: 0, 1: 1, 2: 1, 3: 3},
        input_channel_dim_to_size={3: 64},
    )
    result = rocm_common.get_padding_conv_sizes(
        bounds=[128, 56, 56, 64],
        padding_sizes=[256, 64, 128],
        igemm_loop_iterators=['"parallel"', '"parallel"', '"parallel"'],
        conv_to_igemm_info=conv_to_igemm_info,
    )
    assert result is None

    conv_to_igemm_info = common.ConvToIgemmInfo(
        is_batch_dim_last=False,
        is_spatial_dim_last=False,
        conv_dims=SimpleNamespace(batch=[0], input_channel=[2]),
        conv_to_igemm_dim_map={0: 0, 1: 1, 2: 3},
        input_channel_dim_to_size={2: 64},
    )
    result = rocm_common.get_padding_conv_sizes(
        bounds=[128, 56, 56, 64],
        padding_sizes=[256, 64, 64, 128],
        igemm_loop_iterators=['"parallel"', '"parallel"', '"parallel"', '"reduction"'],
        conv_to_igemm_info=conv_to_igemm_info,
    )
    assert result is None

    # TODO(Bangtian): Use this programmatic method to create operations instead of
    # textual MLIR strings to fully cleanup the tests.
    # Test with realistic convolution data from IGEMM.
    conv_op = None
    with ir.Location.unknown(tuner_ctx.mlir_ctx):
        f16 = ir.F16Type.get()
        f32 = ir.F32Type.get()

        input_type = ir.RankedTensorType.get([2, 34, 34, 16], f16)
        filter_type = ir.RankedTensorType.get([3, 3, 16, 32], f16)
        output_type = ir.RankedTensorType.get([2, 32, 32, 32], f32)

        @func.FuncOp.from_py_func(input_type, filter_type, output_type)
        def conv_fn(input, filter, output):
            nonlocal conv_op
            result = linalg.conv_2d_nhwc_hwcf(input, filter, outs=[output])
            conv_op = result.owner

    assert conv_op is not None
    convolution_dims = linalg.infer_convolution_dimensions(conv_op)
    igemm_details = iree_codegen.get_igemm_generic_conv_details(conv_op)

    input_type = conv_op.operands[0].type
    res_maps = linalg.get_indexing_maps(conv_op)
    indexing_maps = [map_attr.value for map_attr in res_maps]
    input_map = indexing_maps[0]

    conv_to_igemm_info = dispatch_parser.build_conv_to_igemm_info(
        convolution_dims, input_type, input_map, igemm_details
    )
    assert conv_to_igemm_info is not None
    assert conv_to_igemm_info.is_spatial_dim_last == False
    assert conv_to_igemm_info.is_batch_dim_last == False

    bounds = list(igemm_details.igemm_loop_bounds)
    assert bounds == [2, 32, 32, 32, 144]

    padding_sizes = [4, 64, 64, 64, 256]
    igemm_iterator_types = [str(it) for it in igemm_details.igemm_loop_iterators]
    assert igemm_iterator_types == [
        '"parallel"',
        '"parallel"',
        '"parallel"',
        '"parallel"',
        '"reduction"',
    ]

    result = rocm_common.get_padding_conv_sizes(
        bounds,
        padding_sizes,
        igemm_iterator_types,
        conv_to_igemm_info,
    )
    assert result == [4, 64, 64, 64, 0, 0, 0]
