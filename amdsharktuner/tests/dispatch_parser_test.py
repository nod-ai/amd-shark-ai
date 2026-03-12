# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest dispatch_parser_test.py
"""

# TODO: remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
import amdsharktuner

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import func, iree_codegen, iree_gpu, linalg, scf  # type: ignore

from amdsharktuner import common, dispatch_parser
from amdsharktuner.rocm import rocm_common

from tests.test_utils import (
    build_attention_module,
    build_func_with_generic_contraction,
    build_func_with_matmul_in_forall,
    build_func_with_simple_op,
    build_module_with_constant,
)


def test_get_contraction_operation(tuner_ctx: common.TunerContext) -> None:
    # Test 1: transpose_b matmul
    with ir.Location.unknown():
        module = ir.Module.create()
        dim_m = ir.AffineDimExpr.get(0)
        dim_n = ir.AffineDimExpr.get(1)
        dim_k = ir.AffineDimExpr.get(2)
        build_func_with_generic_contraction(
            module,
            lhs_shape=[16, 64],
            rhs_shape=[32, 64],
            res_shape=[16, 32],
            lhs_map=ir.AffineMap.get(3, 0, [dim_m, dim_k]),
            rhs_map=ir.AffineMap.get(3, 0, [dim_n, dim_k]),
            res_map=ir.AffineMap.get(3, 0, [dim_m, dim_n]),
            iterator_types=["parallel", "parallel", "reduction"],
        )
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)

    # Test 2: bmm with transposed inputs
    with ir.Location.unknown():
        module = ir.Module.create()
        d0 = ir.AffineDimExpr.get(0)
        d1 = ir.AffineDimExpr.get(1)
        d2 = ir.AffineDimExpr.get(2)
        d3 = ir.AffineDimExpr.get(3)
        build_func_with_generic_contraction(
            module,
            lhs_shape=[5, 8, 128],
            rhs_shape=[128, 40, 5],
            res_shape=[5, 40, 8],
            lhs_map=ir.AffineMap.get(4, 0, [d0, d1, d3]),
            rhs_map=ir.AffineMap.get(4, 0, [d3, d2, d0]),
            res_map=ir.AffineMap.get(4, 0, [d0, d2, d1]),
            iterator_types=["parallel", "parallel", "parallel", "reduction"],
        )
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)

    # Test 3: high-rank bmm with transposed inputs
    with ir.Location.unknown():
        module = ir.Module.create()
        b0 = ir.AffineDimExpr.get(0)
        m0 = ir.AffineDimExpr.get(1)
        n0 = ir.AffineDimExpr.get(2)
        k0 = ir.AffineDimExpr.get(3)
        b1 = ir.AffineDimExpr.get(4)
        m1 = ir.AffineDimExpr.get(5)
        n1 = ir.AffineDimExpr.get(6)
        k1 = ir.AffineDimExpr.get(7)
        build_func_with_generic_contraction(
            module,
            lhs_shape=[16, 8, 15, 16, 64, 256],
            rhs_shape=[16, 9, 15, 16, 128, 256],
            res_shape=[16, 8, 9, 16, 64, 128],
            lhs_map=ir.AffineMap.get(8, 0, [b0, m0, k0, b1, m1, k1]),
            rhs_map=ir.AffineMap.get(8, 0, [b0, n0, k0, b1, n1, k1]),
            res_map=ir.AffineMap.get(8, 0, [b0, m0, n0, b1, m1, n1]),
            iterator_types=[
                "parallel",
                "parallel",
                "parallel",
                "reduction",
                "parallel",
                "parallel",
                "parallel",
                "reduction",
            ],
        )
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]
    parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
    assert dispatch_parser.get_parent_function_name(parser.get_root_op()) == "test"


def test_get_matmul_named_op(tuner_ctx: common.TunerContext) -> None:
    with ir.Location.unknown(tuner_ctx.mlir_ctx):
        module = ir.Module.create()
        f16 = ir.F16Type.get()
        f32 = ir.F32Type.get()

        with ir.InsertionPoint(module.body):
            a_type = ir.RankedTensorType.get((16, 64), f16)
            b_type = ir.RankedTensorType.get((64, 32), f16)
            c_type = ir.RankedTensorType.get((16, 32), f32)

            dim_m = ir.AffineDimExpr.get(0)
            dim_n = ir.AffineDimExpr.get(1)
            dim_k = ir.AffineDimExpr.get(2)
            a_map = ir.AffineMap.get(3, 0, [dim_m, dim_k])
            b_map = ir.AffineMap.get(3, 0, [dim_k, dim_n])
            c_map = ir.AffineMap.get(3, 0, [dim_m, dim_n])

            @func.FuncOp.from_py_func(a_type, b_type, c_type)
            def named_matmul(a, b, c):
                matmul_op = linalg.MatmulOp(
                    result_tensors=[c_type],
                    inputs=[a, b],
                    outputs=[c],
                    indexing_maps=[a_map, b_map, c_map],
                )
                matmul_op.operation.attributes[
                    "root_op"
                ] = iree_codegen.RootOpAttr.get()

        root_op_list = iree_codegen.get_tuner_root_ops(module)
        assert len(root_op_list) == 1, "Expected one root op"
        root_op = root_op_list[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        assert (
            dispatch_parser.get_parent_function_name(parser.get_root_op())
            == "named_matmul"
        )


def test_get_named_contraction_op(tuner_ctx: common.TunerContext):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        f32 = ir.F32Type.get()

        with ir.InsertionPoint(module.body):
            lhs_type = ir.RankedTensorType.get((5, 3), f32)
            rhs_type = ir.RankedTensorType.get((7, 3), f32)
            res_type = ir.RankedTensorType.get((5, 7), f32)

            @func.FuncOp.from_py_func(lhs_type, rhs_type, res_type)
            def named_contraction(lhs, rhs, res):
                dim_i = ir.AffineDimExpr.get(0)
                dim_j = ir.AffineDimExpr.get(1)
                dim_k = ir.AffineDimExpr.get(2)

                lhs_map = ir.AffineMap.get(3, 0, [dim_i, dim_k])
                rhs_map = ir.AffineMap.get(3, 0, [dim_j, dim_k])
                res_map = ir.AffineMap.get(3, 0, [dim_i, dim_j])

                contraction_op = linalg.ContractOp(
                    result_tensors=[res_type],
                    inputs=[lhs, rhs],
                    outputs=[res],
                    indexing_maps=[lhs_map, rhs_map, res_map],
                )
                contraction_op.attributes["root_op"] = iree_codegen.RootOpAttr.get()

        root_op_list = iree_codegen.get_tuner_root_ops(module)
        assert len(root_op_list) == 1
        root_op = root_op_list[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        assert (
            dispatch_parser.get_parent_function_name(parser.get_root_op())
            == "named_contraction"
        )


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


def test_parse_mlir(tuner_ctx: common.TunerContext) -> None:
    with ir.Location.unknown():
        module = ir.Module.create()
        build_func_with_simple_op(module, func_name="simple_mul")
    mlir_str = str(module)

    mlir_module = dispatch_parser.parse_mlir(mlir_str, tuner_ctx)
    assert mlir_module is not None
    assert isinstance(mlir_module, ir.Module)
    assert isinstance(mlir_module.body.operations[0], func.FuncOp)


def test_get_attention_operation(tuner_ctx: common.TunerContext) -> None:
    module = build_attention_module(
        batch=20,
        seq_len=4096,
        head_dim=64,
        func_name="attention_20x4096x64x4096x64",
    )
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]

    parser = dispatch_parser.AttentionOpInterfaceParser(root_op, tuner_ctx)
    assert (
        dispatch_parser.get_parent_function_name(parser.get_root_op())
        == "attention_20x4096x64x4096x64"
    )

    indexing_maps_attr = root_op.attributes["indexing_maps"]
    affine_maps = [attr.value for attr in indexing_maps_attr]
    q_map = affine_maps[0]
    k_map = affine_maps[1]
    v_map = affine_maps[2]
    o_map = affine_maps[-1]
    result = iree_codegen.get_attention_op_detail(q_map, k_map, v_map, o_map)

    assert result.domain_rank == 5
    assert result.batch_dims == [0]
    assert result.m_dims == [1]
    assert result.k1_dims == [2]
    assert result.k2_dims == [3]
    assert result.n_dims == [4]


def test_get_parent_function_name_with_split_reduction(
    tuner_ctx: common.TunerContext,
) -> None:
    """Test get_parent_function_name when root op is nested in scf.forall (split reduction)."""
    f16 = ir.F16Type.get()
    f32 = ir.F32Type.get()
    with ir.Location.unknown():
        module = ir.Module.create()
        build_func_with_matmul_in_forall(
            module,
            m=16,
            n=32,
            k=64,
            num_threads=4,
            lhs_type=f16,
            rhs_type=f16,
            res_type=f32,
            func_name="split_reduction_test",
        )
    root_op_list = iree_codegen.get_tuner_root_ops(module)
    assert len(root_op_list) == 1
    root_op = root_op_list[0]

    # The root op is nested: func.func -> scf.forall -> linalg.matmul.
    # get_parent_function_name should walk up the hierarchy and find the function.
    func_name = dispatch_parser.get_parent_function_name(root_op)
    assert func_name == "split_reduction_test"

    parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
    assert (
        dispatch_parser.get_parent_function_name(parser.get_root_op())
        == "split_reduction_test"
    )


def test_get_parent_function_name_no_function(
    tuner_ctx: common.TunerContext,
) -> None:
    """Test get_parent_function_name returns None when no func.func is found."""
    with ir.Location.unknown():
        module = ir.Module.create()
        constant_op = build_module_with_constant(module)

    func_name = dispatch_parser.get_parent_function_name(constant_op)
    assert func_name is None
