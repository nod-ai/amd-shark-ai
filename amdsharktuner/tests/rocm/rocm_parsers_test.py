# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
import amdsharktuner

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, linalg  # type: ignore

from amdsharktuner import common, dispatch_parser
from amdsharktuner.rocm import rocm_parsers
from amdsharktuner.test_utils import (
    build_func_with_conv2d_nhwc_hwcf,
    build_func_with_generic_contraction,
    build_func_with_group_conv,
    tuner_ctx,
)


def test_build_conv_to_igemm_info(tuner_ctx: common.TunerContext) -> None:
    f16 = ir.F16Type.get()
    f32 = ir.F32Type.get()
    with ir.Location.unknown():
        module = ir.Module.create()
        build_func_with_conv2d_nhwc_hwcf(
            module,
            input_shape=(2, 34, 34, 16),
            filter_shape=(3, 3, 16, 32),
            output_shape=(2, 32, 32, 32),
            input_type=f16,
            filter_type=f16,
            output_type=f32,
        )
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]

    convolution_dims = linalg.infer_convolution_dimensions(root_op)
    assert convolution_dims is not None

    igemm_details = iree_codegen.get_igemm_generic_conv_details(root_op)
    assert igemm_details is not None

    input_type = root_op.operands[0].type
    res_maps = linalg.get_indexing_maps(root_op)
    indexing_maps = [map_attr.value for map_attr in res_maps]
    input_map = indexing_maps[0]

    conv_to_igemm_info = rocm_parsers.build_conv_to_igemm_info(
        convolution_dims, input_type, input_map, igemm_details
    )

    assert conv_to_igemm_info is not None

    # NHWC layout: spatial and batch dims are not last.
    assert conv_to_igemm_info.is_spatial_dim_last == False
    assert conv_to_igemm_info.is_batch_dim_last == False

    assert conv_to_igemm_info.conv_to_igemm_dim == igemm_details.conv_to_igemm_dim_map
    assert conv_to_igemm_info.conv_to_igemm_dim == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 4,
        6: 4,
    }

    assert conv_to_igemm_info.input_channel_dim_to_size == {6: 16}
    assert list(convolution_dims.input_channel) == [6]
    assert input_type.shape[3] == 16


def test_get_conv_nhwc_hwcf_operation(tuner_ctx: common.TunerContext) -> None:
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    with ir.Location.unknown():
        module = ir.Module.create()
        build_func_with_conv2d_nhwc_hwcf(
            module,
            input_shape=(2, 34, 34, 16),
            filter_shape=(3, 3, 16, 16),
            output_shape=(2, 32, 32, 16),
            input_type=i8,
            filter_type=i8,
            output_type=i32,
            with_fill=True,
        )
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"


def test_get_group_conv_operation(tuner_ctx: common.TunerContext) -> None:
    with ir.Location.unknown():
        module = ir.Module.create()
        build_func_with_group_conv(
            module,
            input_shape=[2, 10, 10, 7, 4],
            filter_shape=[7, 16, 3, 3, 4],
            output_shape=[2, 8, 8, 7, 16],
        )
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"


def test_get_generic_conv_operation(tuner_ctx: common.TunerContext) -> None:
    with ir.Location.name("generic_conv"):
        # nhwc_hwcf
        module = ir.Module.create()
        d0 = ir.AffineDimExpr.get(0)
        d1 = ir.AffineDimExpr.get(1)
        d2 = ir.AffineDimExpr.get(2)
        d3 = ir.AffineDimExpr.get(3)
        d4 = ir.AffineDimExpr.get(4)
        d5 = ir.AffineDimExpr.get(5)
        d6 = ir.AffineDimExpr.get(6)
        build_func_with_generic_contraction(
            module,
            lhs_shape=[2, 7, 7, 32],
            rhs_shape=[3, 3, 32, 64],
            res_shape=[2, 5, 5, 64],
            lhs_map=ir.AffineMap.get(7, 0, [d0, d1 + d4, d2 + d5, d6]),
            rhs_map=ir.AffineMap.get(7, 0, [d4, d5, d6, d3]),
            res_map=ir.AffineMap.get(7, 0, [d0, d1, d2, d3]),
            iterator_types=[
                "parallel",
                "parallel",
                "parallel",
                "parallel",
                "reduction",
                "reduction",
                "reduction",
            ],
        )
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"
