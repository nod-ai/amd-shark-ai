# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import pytest

from typing import Generator
from logging import Logger
from unittest.mock import MagicMock

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import arith, func, iree_codegen, linalg, scf, tensor  # type: ignore

from amdsharktuner import common


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    mock_logger = MagicMock(spec=Logger)
    with common.TunerContext(logger=mock_logger) as ctx:
        yield ctx


@pytest.fixture
def mlir_ctx() -> Generator[ir.Context, None, None]:
    with ir.Context() as ctx:
        yield ctx


# -----------------------------------------------------------------------------
# Matmul builders
# -----------------------------------------------------------------------------


def build_func_with_matmul(
    module: ir.Module,
    m: int,
    n: int,
    k: int,
    lhs_type: ir.Type,
    rhs_type: ir.Type,
    res_type: ir.Type,
) -> None:
    """Build a func containing a linalg.matmul op."""
    a_type = ir.RankedTensorType.get((m, k), lhs_type)
    b_type = ir.RankedTensorType.get((k, n), rhs_type)
    c_type = ir.RankedTensorType.get((m, n), res_type)

    dim_m = ir.AffineDimExpr.get(0)
    dim_n = ir.AffineDimExpr.get(1)
    dim_k = ir.AffineDimExpr.get(2)
    a_map = ir.AffineMap.get(3, 0, [dim_m, dim_k])
    b_map = ir.AffineMap.get(3, 0, [dim_k, dim_n])
    c_map = ir.AffineMap.get(3, 0, [dim_m, dim_n])

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(a_type, b_type, c_type)
        def named_matmul(a: ir.Value, b: ir.Value, c: ir.Value) -> None:
            matmul_op = linalg.MatmulOp(
                result_tensors=[c_type],
                inputs=[a, b],
                outputs=[c],
                indexing_maps=[a_map, b_map, c_map],
            )
            matmul_op.operation.attributes["root_op"] = iree_codegen.RootOpAttr.get()


def build_func_with_matmul_in_forall(
    module: ir.Module,
    m: int,
    n: int,
    k: int,
    num_threads: int,
    lhs_type: ir.Type,
    rhs_type: ir.Type,
    res_type: ir.Type,
    func_name: str = "test",
) -> None:
    """Build a func containing a linalg.matmul nested inside scf.forall."""
    lhs_tensor_type = ir.RankedTensorType.get((m, k), lhs_type)
    rhs_tensor_type = ir.RankedTensorType.get((k, n), rhs_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(lhs_tensor_type, rhs_tensor_type, name=func_name)
        def forall_matmul(lhs, rhs):
            init = tensor.empty([m, n], res_type)
            forall_op = scf.ForallOp(
                lower_bounds=[0],
                upper_bounds=[num_threads],
                steps=[1],
                shared_outs=[init],
            )

            with ir.InsertionPoint(forall_op.body):
                out_arg = forall_op.body.arguments[-1]
                matmul_op = linalg.matmul(lhs, rhs, outs=[out_arg])
                matmul_op.owner.attributes["root_op"] = iree_codegen.RootOpAttr.get()

                in_parallel_op = scf.InParallelOp()
                with ir.InsertionPoint(in_parallel_op.region.blocks[0]):
                    tensor.ParallelInsertSliceOp(
                        source=matmul_op,
                        dest=out_arg,
                        offsets=[],
                        sizes=[],
                        strides=[],
                        static_offsets=[0, 0],
                        static_sizes=[m, n],
                        static_strides=[1, 1],
                    )

            return forall_op.results[0]


# -----------------------------------------------------------------------------
# Generic contraction builder
# -----------------------------------------------------------------------------


def build_func_with_generic_contraction(
    module: ir.Module,
    lhs_shape: list[int],
    rhs_shape: list[int],
    res_shape: list[int],
    lhs_map: ir.AffineMap,
    rhs_map: ir.AffineMap,
    res_map: ir.AffineMap,
    iterator_types: list[str],
    lhs_type: ir.Type | None = None,
    rhs_type: ir.Type | None = None,
    res_type: ir.Type | None = None,
) -> None:
    """Build a func containing a linalg.generic contraction op."""
    f16 = ir.F16Type.get()
    f32 = ir.F32Type.get()

    lhs_elem_type = lhs_type if lhs_type is not None else f16
    rhs_elem_type = rhs_type if rhs_type is not None else f16
    res_elem_type = res_type if res_type is not None else f32

    lhs_tensor_type = ir.RankedTensorType.get(lhs_shape, lhs_elem_type)
    rhs_tensor_type = ir.RankedTensorType.get(rhs_shape, rhs_elem_type)
    res_tensor_type = ir.RankedTensorType.get(res_shape, res_elem_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(lhs_tensor_type, rhs_tensor_type)
        def test(lhs, rhs):
            cst = arith.constant(f32, 0.0)
            empty = tensor.empty(res_shape, res_elem_type)
            filled = linalg.fill(cst, outs=[empty])

            generic_op = linalg.GenericOp(
                result_tensors=[res_tensor_type],
                inputs=[lhs, rhs],
                outputs=[filled],
                indexing_maps=[lhs_map, rhs_map, res_map],
                iterator_types=iterator_types,
            )
            generic_op.operation.attributes["root_op"] = iree_codegen.RootOpAttr.get()

            block = generic_op.regions[0].blocks.append(
                lhs_elem_type, rhs_elem_type, res_elem_type
            )
            with ir.InsertionPoint(block):
                in0, in1, out = block.arguments
                ext0 = arith.ExtFOp(f32, in0).result
                ext1 = arith.ExtFOp(f32, in1).result
                mul = arith.MulFOp(ext0, ext1).result
                add = arith.AddFOp(out, mul).result
                linalg.YieldOp([add])

            return generic_op.result


# -----------------------------------------------------------------------------
# Convolution builders
# -----------------------------------------------------------------------------


def build_func_with_conv2d_nhwc_hwcf(
    module: ir.Module,
    input_shape: tuple[int, int, int, int] | list[int],
    filter_shape: tuple[int, int, int, int] | list[int],
    output_shape: tuple[int, int, int, int] | list[int],
    input_type: ir.Type,
    filter_type: ir.Type,
    output_type: ir.Type,
    strides: tuple[int, int] | None = None,
    dilations: tuple[int, int] | None = None,
    with_fill: bool = False,
    func_name: str = "test",
) -> None:
    """Build a func containing a linalg.conv_2d_nhwc_hwcf op."""
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    filter_tensor_type = ir.RankedTensorType.get(filter_shape, filter_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):
        if with_fill:

            @func.FuncOp.from_py_func(
                input_tensor_type, filter_tensor_type, name=func_name
            )
            def conv_func(inp, filt):
                if output_type == ir.IntegerType.get_signless(32):
                    cst = arith.constant(output_type, 0)
                else:
                    cst = arith.constant(output_type, 0.0)
                empty = tensor.empty(list(output_shape), output_type)
                filled = linalg.fill(cst, outs=[empty])
                conv_op = linalg.conv_2d_nhwc_hwcf(inp, filt, outs=[filled])
                conv_op.owner.attributes["root_op"] = iree_codegen.RootOpAttr.get()
                return conv_op

            return

        @func.FuncOp.from_py_func(
            input_tensor_type,
            filter_tensor_type,
            output_tensor_type,
            name=func_name,
        )
        def conv_func(arg0, arg1, arg2):
            strides_attr = ir.DenseI64ArrayAttr.get(list(strides)) if strides else None
            dilations_attr = (
                ir.DenseI64ArrayAttr.get(list(dilations)) if dilations else None
            )
            conv_op = linalg.Conv2DNhwcHwcfOp(
                inputs=[arg0, arg1],
                outputs=[arg2],
                result_tensors=[output_tensor_type],
                strides=strides_attr,
                dilations=dilations_attr,
            )
            conv_op.operation.attributes["root_op"] = iree_codegen.RootOpAttr.get()


def build_func_with_conv2d_nchw_fchw(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    filter_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    filter_type: ir.Type,
    output_type: ir.Type,
) -> None:
    """Build a func containing a linalg.conv_2d_nchw_fchw op."""
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    filter_tensor_type = ir.RankedTensorType.get(filter_shape, filter_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, filter_tensor_type, output_tensor_type
        )
        def conv2d_func(arg0, arg1, arg2):
            conv_op = linalg.Conv2DNchwFchwOp(
                inputs=[arg0, arg1],
                outputs=[arg2],
                result_tensors=[output_tensor_type],
            )
            # Explicitly set unit strides and dilations as MLIR attributes.
            conv_op.operation.attributes["strides"] = ir.DenseI64ArrayAttr.get([1, 1])
            conv_op.operation.attributes["dilations"] = ir.DenseI64ArrayAttr.get([1, 1])
            conv_op.operation.attributes["root_op"] = iree_codegen.RootOpAttr.get()


def build_func_with_conv2d_nhwc_fhwc(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    filter_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    filter_type: ir.Type,
    output_type: ir.Type,
) -> None:
    """Build a func containing a linalg.generic conv with NHWC x FHWC layout."""
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    filter_tensor_type = ir.RankedTensorType.get(filter_shape, filter_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, filter_tensor_type, output_tensor_type
        )
        def conv2d_func(arg0, arg1, arg2):
            # NHWC x FHWC: (d0, d1, d2, d3, d4, d5, d6).
            # input: (d0, d1+d4, d2+d5, d6) - (N, H+kH, W+kW, C).
            # filter: (d3, d4, d5, d6) - (F, kH, kW, C).
            # output: (d0, d1, d2, d3) - (N, H, W, F).
            d0, d1, d2, d3, d4, d5, d6 = [ir.AffineDimExpr.get(i) for i in range(7)]
            input_map = ir.AffineMap.get(7, 0, [d0, d1 + d4, d2 + d5, d6])
            filter_map = ir.AffineMap.get(7, 0, [d3, d4, d5, d6])
            output_map = ir.AffineMap.get(7, 0, [d0, d1, d2, d3])

            generic_op = linalg.GenericOp(
                result_tensors=[output_tensor_type],
                inputs=[arg0, arg1],
                outputs=[arg2],
                indexing_maps=[input_map, filter_map, output_map],
                iterator_types=[
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.reduction,
                    linalg.IteratorType.reduction,
                    linalg.IteratorType.reduction,
                ],
            )
            block = generic_op.regions[0].blocks.append(
                input_type, filter_type, output_type
            )
            with ir.InsertionPoint(block):
                in0 = arith.ExtFOp(output_type, block.arguments[0]).result
                in1 = arith.ExtFOp(output_type, block.arguments[1]).result
                mul = arith.MulFOp(in0, in1).result
                add = arith.AddFOp(block.arguments[2], mul).result
                linalg.YieldOp([add])
            generic_op.operation.attributes["root_op"] = iree_codegen.RootOpAttr.get()


def build_func_with_conv2d_chwn_chwf(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    filter_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    filter_type: ir.Type,
    output_type: ir.Type,
) -> None:
    """Build a func containing a linalg.generic conv with CHWN x CHWF layout."""
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    filter_tensor_type = ir.RankedTensorType.get(filter_shape, filter_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, filter_tensor_type, output_tensor_type
        )
        def conv2d_func(arg0, arg1, arg2):
            # CHWN x CHWF: (d0, d1, d2, d3, d4, d5, d6).
            # d0=F, d1=oH, d2=oW, d3=N (parallel).
            # d4=C, d5=kH, d6=kW (reduction).
            # input: (d4, d1+d5, d2+d6, d3) - (C, H, W, N).
            # filter: (d4, d5, d6, d0) - (C, kH, kW, F).
            # output: (d0, d1, d2, d3) - (F, oH, oW, N).
            d0, d1, d2, d3, d4, d5, d6 = [ir.AffineDimExpr.get(i) for i in range(7)]
            input_map = ir.AffineMap.get(7, 0, [d4, d1 + d5, d2 + d6, d3])
            filter_map = ir.AffineMap.get(7, 0, [d4, d5, d6, d0])
            output_map = ir.AffineMap.get(7, 0, [d0, d1, d2, d3])

            generic_op = linalg.GenericOp(
                result_tensors=[output_tensor_type],
                inputs=[arg0, arg1],
                outputs=[arg2],
                indexing_maps=[input_map, filter_map, output_map],
                iterator_types=[
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.reduction,
                    linalg.IteratorType.reduction,
                    linalg.IteratorType.reduction,
                ],
            )
            block = generic_op.regions[0].blocks.append(
                input_type, filter_type, output_type
            )
            with ir.InsertionPoint(block):
                in0 = arith.ExtFOp(output_type, block.arguments[0]).result
                in1 = arith.ExtFOp(output_type, block.arguments[1]).result
                mul = arith.MulFOp(in0, in1).result
                add = arith.AddFOp(block.arguments[2], mul).result
                linalg.YieldOp([add])
            generic_op.operation.attributes["root_op"] = iree_codegen.RootOpAttr.get()


def build_func_with_conv2d_nhwc_hwcf_dynamic(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    filter_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    filter_type: ir.Type,
    output_type: ir.Type,
    dynamic_dims: list[int] | None = None,
) -> None:
    """Build a linalg.conv_2d_nhwc_hwcf op with optional dynamic dimensions."""
    actual_output_shape: tuple[int, ...] = output_shape
    if dynamic_dims:
        output_shape_list = list(output_shape)
        for dim in dynamic_dims:
            output_shape_list[dim] = ir.ShapedType.get_dynamic_size()
        actual_output_shape = tuple(output_shape_list)

    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    filter_tensor_type = ir.RankedTensorType.get(filter_shape, filter_type)
    output_tensor_type = ir.RankedTensorType.get(actual_output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, filter_tensor_type, output_tensor_type
        )
        def conv2d_func(arg0, arg1, arg2):
            conv_op = linalg.Conv2DNhwcHwcfOp(
                inputs=[arg0, arg1],
                outputs=[arg2],
                result_tensors=[output_tensor_type],
            )
            conv_op.operation.attributes["root_op"] = iree_codegen.RootOpAttr.get()


# -----------------------------------------------------------------------------
# Group convolution builders
# -----------------------------------------------------------------------------


def build_func_with_group_conv(
    module: ir.Module,
    input_shape: list[int],
    filter_shape: list[int],
    output_shape: list[int],
    strides: list[int] | None = None,
    dilations: list[int] | None = None,
    elem_type: ir.Type | None = None,
    func_name: str = "test",
) -> None:
    """Build a func containing a linalg.conv_2d_nhwgc_gfhwc (group conv) op."""
    f32 = ir.F32Type.get()
    elem = elem_type if elem_type is not None else f32
    strides_val = strides if strides is not None else [1, 1]
    dilations_val = dilations if dilations is not None else [1, 1]

    input_tensor = ir.RankedTensorType.get(input_shape, elem)
    filter_tensor = ir.RankedTensorType.get(filter_shape, elem)
    output_tensor = ir.RankedTensorType.get(output_shape, elem)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor, filter_tensor, output_tensor, name=func_name
        )
        def group_conv_func(inp, filt, out):
            conv_op = linalg.conv_2d_nhwgc_gfhwc(
                inp, filt, outs=[out], strides=strides_val, dilations=dilations_val
            )
            conv_op.owner.attributes["root_op"] = iree_codegen.RootOpAttr.get()
            return conv_op


def build_func_with_group_conv2d_nhwgc_gfhwc(
    module: ir.Module,
    input_shape: tuple[int, int, int, int, int],
    filter_shape: tuple[int, int, int, int, int],
    output_shape: tuple[int, int, int, int, int],
    input_type: ir.Type,
    filter_type: ir.Type,
    output_type: ir.Type,
) -> None:
    """Build a func containing a linalg.generic group conv with NHWGC x GFHWC layout."""
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    filter_tensor_type = ir.RankedTensorType.get(filter_shape, filter_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, filter_tensor_type, output_tensor_type
        )
        def group_conv_func(arg0, arg1, arg2):
            # Group conv: (d0, d1, d2, d3, d4, d5, d6, d7).
            # input:  (d0, d1+d5, d2+d6, d3, d7) - (N, H+kH, W+kW, G, C).
            # filter: (d3, d4, d5, d6, d7) - (G, F, kH, kW, C).
            # output: (d0, d1, d2, d3, d4) - (N, H, W, G, F).
            d0, d1, d2, d3, d4, d5, d6, d7 = [ir.AffineDimExpr.get(i) for i in range(8)]
            input_map = ir.AffineMap.get(8, 0, [d0, d1 + d5, d2 + d6, d3, d7])
            filter_map = ir.AffineMap.get(8, 0, [d3, d4, d5, d6, d7])
            output_map = ir.AffineMap.get(8, 0, [d0, d1, d2, d3, d4])

            generic_op = linalg.GenericOp(
                result_tensors=[output_tensor_type],
                inputs=[arg0, arg1],
                outputs=[arg2],
                indexing_maps=[input_map, filter_map, output_map],
                iterator_types=[
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.parallel,
                    linalg.IteratorType.reduction,
                    linalg.IteratorType.reduction,
                    linalg.IteratorType.reduction,
                ],
            )
            block = generic_op.regions[0].blocks.append(
                input_type, filter_type, output_type
            )
            with ir.InsertionPoint(block):
                in0 = arith.ExtFOp(output_type, block.arguments[0]).result
                in1 = arith.ExtFOp(output_type, block.arguments[1]).result
                mul = arith.MulFOp(in0, in1).result
                add = arith.AddFOp(block.arguments[2], mul).result
                linalg.YieldOp([add])

            generic_op.operation.attributes["root_op"] = iree_codegen.RootOpAttr.get()


# -----------------------------------------------------------------------------
# Attention builder
# -----------------------------------------------------------------------------


def build_attention_module(
    batch: int,
    seq_len: int,
    head_dim: int,
    kv_seq_len: int | None = None,
    elem_type: str = "f16",
    func_name: str = "test",
) -> ir.Module:
    """Build a module containing an iree_linalg_ext.attention op."""
    kv_len = kv_seq_len if kv_seq_len is not None else seq_len

    q_type = f"tensor<{batch}x{seq_len}x{head_dim}x{elem_type}>"
    k_type = f"tensor<{batch}x{kv_len}x{head_dim}x{elem_type}>"
    v_type = f"tensor<{batch}x{kv_len}x{head_dim}x{elem_type}>"
    o_type = f"tensor<{batch}x{seq_len}x{head_dim}x{elem_type}>"

    module_str = f"""
    builtin.module {{
        func.func @{func_name}(
            %q : {q_type},
            %k : {k_type},
            %v : {v_type},
            %scale : {elem_type},
            %output : {o_type}
        ) -> {o_type} {{
            %result = iree_linalg_ext.attention {{
                root_op = #iree_codegen.root_op<set = 0>,
                indexing_maps = [
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                    affine_map<(d0, d1, d2, d3, d4) -> ()>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
                ]
            }} ins(%q, %k, %v, %scale : {q_type}, {k_type}, {v_type}, {elem_type})
              outs(%output : {o_type}) {{
            ^bb0(%score: f32):
                iree_linalg_ext.yield %score : f32
            }} -> {o_type}
            return %result : {o_type}
        }}
    }}
    """
    return ir.Module.parse(module_str)


# -----------------------------------------------------------------------------
# Simple/utility builders
# -----------------------------------------------------------------------------


def build_func_with_simple_op(
    module: ir.Module,
    func_name: str = "test",
) -> None:
    """Build a func containing a simple arith.mulf op."""
    f32 = ir.F32Type.get()
    tensor_type = ir.RankedTensorType.get([4], f32)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(tensor_type, tensor_type, name=func_name)
        def simple_func(arg0, arg1):
            return arith.MulFOp(arg0, arg1).result


def build_module_with_constant(module: ir.Module) -> ir.Operation:
    """Build a module containing just an arith.constant (no func)."""
    f32 = ir.F32Type.get()
    with ir.InsertionPoint(module.body):
        cst_op = arith.ConstantOp(f32, 0.0)
        return cst_op.operation
