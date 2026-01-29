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
from amdsharktuner.test_utils import tuner_ctx


GENERIC_TEMPLATE = r"""
builtin.module{{
    func.func @test(%arg0: {lhs_type}, %arg1: {rhs_type}) -> {res_type} {{
        %cst = arith.constant 0.000000e+00 : f32
        %0 = tensor.empty() : {res_type}
        %1 = linalg.fill ins(%cst : f32) outs(%0 : {res_type}) -> {res_type}
        %2 = linalg.generic {{
            indexing_maps = [
                {lhs_map},
                {rhs_map},
                {res_map}],
            iterator_types = {iterator_types}}}
            {{root_op}}
            ins(%arg0, %arg1 : {lhs_type}, {rhs_type})
            outs(%1 : {res_type}) {{
        ^bb0(%in: f16, %in_0: f16, %out: f32):
            %3 = arith.extf %in : f16 to f32
            %4 = arith.extf %in_0 : f16 to f32
            %5 = arith.mulf %3, %4 : f32
            %6 = arith.addf %out, %5 : f32
            linalg.yield %6 : f32
        }} -> {res_type}
        return %2 : {res_type}
    }}
}}
"""


def test_build_conv_to_igemm_info(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module {
            func.func @test(%arg0: tensor<2x34x34x16xf16>, %arg1: tensor<3x3x16x32xf16>, %arg2: tensor<2x32x32x32xf32>) -> tensor<2x32x32x32xf32> {
                %0 = linalg.conv_2d_nhwc_hwcf {root_op}
                    ins(%arg0, %arg1 : tensor<2x34x34x16xf16>, tensor<3x3x16x32xf16>)
                    outs(%arg2 : tensor<2x32x32x32xf32>) -> tensor<2x32x32x32xf32>
                return %0 : tensor<2x32x32x32xf32>
            }
        }"""

    module = ir.Module.parse(module_str, context)
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
    assert conv_to_igemm_info.conv_dims == convolution_dims

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
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<2x34x34x16xi8>, %arg1: tensor<3x3x16x16xi8>) -> tensor<2x32x32x16xi32> {
                %cst = arith.constant 0 : i32
                %0 = tensor.empty() : tensor<2x32x32x16xi32>
                %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2x32x32x16xi32>) -> tensor<2x32x32x16xi32>
                %2 = linalg.conv_2d_nhwc_hwcf {root_op}
                    ins(%arg0, %arg1 : tensor<2x34x34x16xi8>, tensor<3x3x16x16xi8>)
                    outs(%1 : tensor<2x32x32x16xi32>) -> tensor<2x32x32x16xi32>
                return %2 : tensor<2x32x32x16xi32>
            }
        }"""
    module = ir.Module.parse(module_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"


def test_get_group_conv_operation(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
    module {
      func.func @test(%arg0: tensor<2x10x10x7x4xf32>, %arg1: tensor<7x16x3x3x4xf32>, %arg2: tensor<2x8x8x7x16xf32>) -> tensor<2x8x8x7x16xf32> {
        %0 = linalg.conv_2d_nhwgc_gfhwc {
           root_op,
           dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>
        } ins(%arg0, %arg1: tensor<2x10x10x7x4xf32>, tensor<7x16x3x3x4xf32>)
          outs(%arg2: tensor<2x8x8x7x16xf32>) -> tensor<2x8x8x7x16xf32>
        return %0 : tensor<2x8x8x7x16xf32>
      }
    }
    """
    module = ir.Module.parse(module_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"


def test_get_generic_conv_operation(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    with ir.Location.name("generic_conv"):
        # nhwc_hwcf
        module_str = GENERIC_TEMPLATE.format(
            lhs_type=ir.RankedTensorType.get([2, 7, 7, 32], ir.F16Type.get()),
            rhs_type=ir.RankedTensorType.get([3, 3, 32, 64], ir.F16Type.get()),
            res_type=ir.RankedTensorType.get([2, 5, 5, 64], ir.F32Type.get()),
            lhs_map="affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>",
            rhs_map="affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>",
            res_map="affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>",
            iterator_types='["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]',
        )
    module = ir.Module.parse(module_str, context)
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"
