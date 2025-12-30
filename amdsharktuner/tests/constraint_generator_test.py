# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest constraint_generator_test.py
"""

import z3  # type: ignore

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import arith, func, iree_codegen, iree_gpu, linalg  # type: ignore

from amdsharktuner import (
    common,
    constraint_generator,
    dispatch_parser,
)
from amdsharktuner.rocm import (
    rocm_constraint_generators,
    rocm_dispatch_constraints,
    rocm_solutions,
)


def build_func_with_matmul(
    module: ir.Module,
    m: int,
    n: int,
    k: int,
    lhs_type: ir.Type,
    rhs_type: ir.Type,
    res_type: ir.Type,
) -> None:
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
            matmul_op.operation.attributes["root_op"] = ir.UnitAttr.get()


def build_func_with_conv2d_nhwc_hwcf(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    kernel_type: ir.Type,
    output_type: ir.Type,
) -> None:
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    kernel_tensor_type = ir.RankedTensorType.get(kernel_shape, kernel_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, kernel_tensor_type, output_tensor_type
        )
        def conv2d_func(arg0, arg1, arg2):
            conv_op = linalg.Conv2DNhwcHwcfOp(
                inputs=[arg0, arg1],
                outputs=[arg2],
                result_tensors=[output_tensor_type],
            )
            conv_op.operation.attributes["root_op"] = ir.UnitAttr.get()


def build_func_with_conv2d_nchw_fchw(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    kernel_type: ir.Type,
    output_type: ir.Type,
) -> None:
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    kernel_tensor_type = ir.RankedTensorType.get(kernel_shape, kernel_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, kernel_tensor_type, output_tensor_type
        )
        def conv2d_func(arg0, arg1, arg2):
            conv_op = linalg.Conv2DNchwFchwOp(
                inputs=[arg0, arg1],
                outputs=[arg2],
                result_tensors=[output_tensor_type],
            )
            conv_op.operation.attributes["root_op"] = ir.UnitAttr.get()


def build_func_with_conv2d_nhwc_fhwc(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    kernel_type: ir.Type,
    output_type: ir.Type,
) -> None:
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    kernel_tensor_type = ir.RankedTensorType.get(kernel_shape, kernel_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, kernel_tensor_type, output_tensor_type
        )
        def conv2d_func(arg0, arg1, arg2):
            # NHWC x FHWC: (d0, d1, d2, d3, d4, d5, d6).
            # input: (d0, d1+d4, d2+d5, d6) - (N, H+kH, W+kW, C).
            # filter: (d3, d4, d5, d6) - (F, kH, kW, C).
            # output: (d0, d1, d2, d3) - (N, H, W, F).
            d0, d1, d2, d3, d4, d5, d6 = [ir.AffineDimExpr.get(i) for i in range(7)]
            input_map = ir.AffineMap.get(7, 0, [d0, d1 + d4, d2 + d5, d6])
            kernel_map = ir.AffineMap.get(7, 0, [d3, d4, d5, d6])
            output_map = ir.AffineMap.get(7, 0, [d0, d1, d2, d3])

            generic_op = linalg.GenericOp(
                result_tensors=[output_tensor_type],
                inputs=[arg0, arg1],
                outputs=[arg2],
                indexing_maps=[input_map, kernel_map, output_map],
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
                input_type, kernel_type, output_type
            )
            with ir.InsertionPoint(block):
                in0 = arith.ExtFOp(output_type, block.arguments[0]).result
                in1 = arith.ExtFOp(output_type, block.arguments[1]).result
                mul = arith.MulFOp(in0, in1).result
                add = arith.AddFOp(block.arguments[2], mul).result
                linalg.YieldOp([add])
            generic_op.operation.attributes["root_op"] = ir.UnitAttr.get()


def build_func_with_conv2d_chwn_chwf(
    module: ir.Module,
    input_shape: tuple[int, int, int, int],
    kernel_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
    input_type: ir.Type,
    kernel_type: ir.Type,
    output_type: ir.Type,
) -> None:
    input_tensor_type = ir.RankedTensorType.get(input_shape, input_type)
    kernel_tensor_type = ir.RankedTensorType.get(kernel_shape, kernel_type)
    output_tensor_type = ir.RankedTensorType.get(output_shape, output_type)

    with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            input_tensor_type, kernel_tensor_type, output_tensor_type
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
            kernel_map = ir.AffineMap.get(7, 0, [d4, d5, d6, d0])
            output_map = ir.AffineMap.get(7, 0, [d0, d1, d2, d3])

            generic_op = linalg.GenericOp(
                result_tensors=[output_tensor_type],
                inputs=[arg0, arg1],
                outputs=[arg2],
                indexing_maps=[input_map, kernel_map, output_map],
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
                input_type, kernel_type, output_type
            )
            with ir.InsertionPoint(block):
                in0 = arith.ExtFOp(output_type, block.arguments[0]).result
                in1 = arith.ExtFOp(output_type, block.arguments[1]).result
                mul = arith.MulFOp(in0, in1).result
                add = arith.AddFOp(block.arguments[2], mul).result
                linalg.YieldOp([add])
            generic_op.operation.attributes["root_op"] = ir.UnitAttr.get()


def test_generate_solutions(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    m, n, k = 2048, 3840, 1280
    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_matmul(module, m, n, k, f16, f16, f32)

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = rocm_constraint_generators.ROCmContractionVectorDistributeConstraintGenerator(
            op_info
        )

        assert gen.op_info.dims.batch == []
        assert gen.op_info.dims.m == [0]
        assert gen.op_info.dims.n == [1]
        assert gen.op_info.dims.k == [2]

        assert gen.op_info.matmul_size.B == []
        assert gen.op_info.matmul_size.M == [2048]
        assert gen.op_info.matmul_size.N == [3840]
        assert gen.op_info.matmul_size.K == [1280]

        assert gen.op_info.lhs_type.shape == [2048, 1280]
        assert gen.op_info.rhs_type.shape == [1280, 3840]
        assert gen.op_info.res_type.shape == [2048, 3840]

        configs = gen.generate_solutions(
            tuner_context=tuner_ctx,
            gpu_target_info=gpu_target_info,
            num_subgroups=4,
            pipeline_options_search_space=rocm_dispatch_constraints.PipelineOptionsSearchSpace(),
        )

        assert list(configs), "Expected at least one valid solution"


def test_generate_attention_solutions(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    op_info = dispatch_parser.AttentionOpInfo(
        root_op=None,
        indexing_maps=[],
        domain_rank=5,
        batch_dims=[0],
        m_dims=[1],
        n_dims=[2],
        k1_dims=[3],
        k2_dims=[4],
        batch_sizes=[2],
        m_sizes=[64],
        n_sizes=[32],
        k1_sizes=[64],
        k2_sizes=[64],
        query_type=f16,
        key_type=f16,
        value_type=f16,
        output_type=f16,
        transposed_q=True,
        transposed_k=True,
        transposed_v=False,
        qk_matmul=common.MatmulShapeType(
            m=64,
            n=64,
            k=64,
            lhs_type=f16,
            rhs_type=f16,
            acc_type=f32,
        ),
        pv_matmul=common.MatmulShapeType(
            m=64,
            n=32,
            k=64,
            lhs_type=f16,
            rhs_type=f16,
            acc_type=f32,
        ),
    )

    solutions = list(
        rocm_solutions.generate_attention_solutions(
            tuner_ctx=tuner_ctx,
            gpu_target_info=gpu_target_info,
            op_info=op_info,
            dispatch_kind=common.DispatchKind.attention,
            num_subgroups=4,
            pipeline_options_search_space=rocm_dispatch_constraints.PipelineOptionsSearchSpace(),
        )
    )

    assert len(solutions) > 0, "Expected at least one valid attention tuning solution"
    for config_list in solutions:
        assert len(config_list) == 2
        assert config_list[0].name == "compilation_info"
        assert config_list[1].name == "decomposition_config"
        assert isinstance(
            config_list[0].configuration, iree_codegen.CompilationInfoAttr
        )
        assert isinstance(config_list[1].configuration, ir.DictAttr)

        # Verify that prefetch_num_stages is set based on layout matching.
        compilation_info = config_list[0].configuration
        translation_info = compilation_info.translation_info
        if translation_info.configuration:
            pipeline_options = translation_info.configuration[
                common.GPU_PIPELINE_OPTIONS_KEY
            ]
            # prefetch_num_stages should be explicitly set to an int (not None).
            assert isinstance(
                pipeline_options.prefetch_num_stages, int
            ), "prefetch_num_stages must be explicitly set to an int"


def test_generate_solutions_tile_and_fuse_contraction_padding(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    m, n, k = 5369, 112, 112

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_matmul(module, m, n, k, f16, f16, f32)

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = rocm_constraint_generators.ROCmContractionTileAndFuseConstraintGenerator(
            op_info
        )

        assert gen.op_info.dims.batch == []
        assert gen.op_info.dims.m == [0]
        assert gen.op_info.dims.n == [1]
        assert gen.op_info.dims.k == [2]

        assert gen.op_info.matmul_size.M == [5369]
        assert gen.op_info.matmul_size.N == [112]
        assert gen.op_info.matmul_size.K == [112]

        assert gen.op_info.lhs_type.shape == [5369, 112]
        assert gen.op_info.rhs_type.shape == [112, 112]
        assert gen.op_info.res_type.shape == [5369, 112]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                num_subgroups=4,
                allowed_waves_per_eu=[2],
                pipeline_options_search_space=rocm_dispatch_constraints.PipelineOptionsSearchSpace(),
            )
        )

        assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
        for solution in solutions:
            assert len(solution) == 1
            config = solution[0]
            assert isinstance(config, common.TuningConfiguration)

            assert config.name == "compilation_info"
            assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)

            lowering_config = config.configuration.lowering_config
            assert "padding =" in str(lowering_config)
            # padding_conv only for convolutions, not contractions.
            assert "padding_conv =" not in str(lowering_config)
            promote = [int(x) for x in lowering_config.attributes["promote_operands"]]
            assert promote == [0, 1]


def test_generate_solutions_tile_and_fuse_conv_padding(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    input_shape = (2, 64, 64, 128)
    kernel_shape = (3, 3, 128, 256)
    output_shape = (2, 62, 62, 256)

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_conv2d_nhwc_hwcf(
            module=module,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            output_shape=output_shape,
            input_type=f16,
            kernel_type=f16,
            output_type=f32,
        )

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.IGEMMConvolutionParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = rocm_constraint_generators.ROCmConvolutionTileAndFuseConstraintGenerator(
            op_info
        )

        # With IGEMM (default), dimensions are restructured to matmul-like form.
        # K dimension is flattened: 3*3*128 = 1152.
        assert gen.op_info.dims.batch == []
        assert gen.op_info.dims.m == [0, 1, 2]
        assert gen.op_info.dims.n == [3]
        assert gen.op_info.dims.k == [4]

        assert gen.op_info.matmul_size.B == []
        assert gen.op_info.matmul_size.M == [2, 62, 62]
        assert gen.op_info.matmul_size.N == [256]
        assert gen.op_info.matmul_size.K == [1152]

        assert gen.op_info.lhs_type.shape == [2, 64, 64, 128]
        assert gen.op_info.rhs_type.shape == [3, 3, 128, 256]
        assert gen.op_info.res_type.shape == [2, 62, 62, 256]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                num_subgroups=4,
            )
        )

        assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
        for solution in solutions:
            assert len(solution) == 1
            config = solution[0]
            assert isinstance(config, common.TuningConfiguration)

            assert config.name == "compilation_info"
            assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)

            lowering_config = config.configuration.lowering_config
            assert "padding =" in str(lowering_config)
            assert "padding_conv =" in str(lowering_config)
            promote = [int(x) for x in lowering_config.attributes["promote_operands"]]
            assert promote == [0, 1]


def test_generate_solutions_tile_and_fuse_conv_small_unaligned(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    input_shape = (2, 7, 7, 32)
    kernel_shape = (3, 3, 32, 64)
    output_shape = (2, 5, 5, 64)

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_conv2d_nhwc_hwcf(
            module=module,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            output_shape=output_shape,
            input_type=f16,
            kernel_type=f16,
            output_type=f32,
        )

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.IGEMMConvolutionParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = rocm_constraint_generators.ROCmConvolutionTileAndFuseConstraintGenerator(
            op_info
        )

        # With IGEMM (default), dimensions are restructured to matmul-like form.
        # K dimension is flattened: 3*3*32 = 288.
        assert gen.op_info.dims.batch == []
        assert gen.op_info.dims.m == [0, 1, 2]
        assert gen.op_info.dims.n == [3]
        assert gen.op_info.dims.k == [4]
        assert gen.op_info.matmul_size.B == []
        assert gen.op_info.matmul_size.M == [2, 5, 5]
        assert gen.op_info.matmul_size.N == [64]
        assert gen.op_info.matmul_size.K == [288]

        assert gen.op_info.lhs_type.shape == [2, 7, 7, 32]
        assert gen.op_info.rhs_type.shape == [3, 3, 32, 64]
        assert gen.op_info.res_type.shape == [2, 5, 5, 64]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                num_subgroups=4,
            )
        )

        assert len(solutions) > 0, "No solutions generated for small unaligned case."

        for solution in solutions:
            assert len(solution) == 1
            config = solution[0]
            assert isinstance(config, common.TuningConfiguration)

            assert config.name == "compilation_info"
            assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)

            lowering_config = config.configuration.lowering_config
            assert "padding =" in str(lowering_config)
            promote = [int(x) for x in lowering_config.attributes["promote_operands"]]
            assert promote == [0, 1]


def test_generate_solutions_tile_and_fuse_conv_nchw_fchw(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    input_shape = (2, 128, 34, 34)
    kernel_shape = (256, 128, 3, 3)
    output_shape = (2, 256, 32, 32)

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_conv2d_nchw_fchw(
            module=module,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            output_shape=output_shape,
            input_type=f16,
            kernel_type=f16,
            output_type=f32,
        )

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.IGEMMConvolutionParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = constraint_generator.ConvolutionOpInterfaceConstraintGenerator(op_info)

        assert gen.op_info.lhs_type.shape == [2, 128, 34, 34]
        assert gen.op_info.rhs_type.shape == [256, 128, 3, 3]
        assert gen.op_info.res_type.shape == [2, 256, 32, 32]
        assert gen.op_info.matmul_size.K == [1152]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
                num_subgroups=4,
            )
        )

        assert len(solutions) > 0, "No solutions generated for NCHW_FCHW conv."
        for solution in solutions:
            assert len(solution) == 1
            config = solution[0]
            assert isinstance(config, common.TuningConfiguration)
            assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)


def test_generate_solutions_tile_and_fuse_conv_nhwc_fhwc(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    input_shape = (2, 10, 10, 32)
    kernel_shape = (64, 3, 3, 32)
    output_shape = (2, 8, 8, 64)

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_conv2d_nhwc_fhwc(
            module=module,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            output_shape=output_shape,
            input_type=f16,
            kernel_type=f16,
            output_type=f32,
        )

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.IGEMMConvolutionParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = constraint_generator.ConvolutionOpInterfaceConstraintGenerator(op_info)

        assert gen.op_info.lhs_type.shape == [2, 10, 10, 32]
        assert gen.op_info.rhs_type.shape == [64, 3, 3, 32]
        assert gen.op_info.res_type.shape == [2, 8, 8, 64]
        assert gen.op_info.matmul_size.K == [288]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
                num_subgroups=4,
            )
        )

        assert len(solutions) > 0, "No solutions generated for NHWC_FHWC conv."
        for solution in solutions:
            assert len(solution) == 1
            config = solution[0]
            assert isinstance(config, common.TuningConfiguration)
            assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)


def test_generate_solutions_tile_and_fuse_conv_chwn_chwf(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    input_shape = (32, 10, 10, 2)
    kernel_shape = (32, 3, 3, 64)
    output_shape = (64, 8, 8, 2)

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_conv2d_chwn_chwf(
            module=module,
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            output_shape=output_shape,
            input_type=f16,
            kernel_type=f16,
            output_type=f32,
        )

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.IGEMMConvolutionParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = constraint_generator.ConvolutionOpInterfaceConstraintGenerator(op_info)

        assert gen.op_info.lhs_type.shape == [32, 10, 10, 2]
        assert gen.op_info.rhs_type.shape == [32, 3, 3, 64]
        assert gen.op_info.res_type.shape == [64, 8, 8, 2]
        assert gen.op_info.matmul_size.K == [288]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
                num_subgroups=4,
            )
        )

        assert len(solutions) > 0, "No solutions generated for CHWN_CHWF conv."
        for solution in solutions:
            assert len(solution) == 1
            config = solution[0]
            assert isinstance(config, common.TuningConfiguration)
            assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)


def test_generate_solutions_tile_and_fuse_matmul_small_unaligned(
    tuner_ctx: common.TunerContext, gpu_target_info: iree_gpu.TargetInfo
) -> None:
    """Test TileAndFuse with small matmul dimensions (< 32) unaligned to intrinsics."""
    context = tuner_ctx.mlir_ctx
    f16 = tuner_ctx.type.f16
    f32 = tuner_ctx.type.f32

    m, n, k = 30, 30, 30

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_matmul(module, m, n, k, f16, f16, f32)

        root_ops = iree_codegen.get_tuner_root_ops(module)
        assert len(root_ops) == 1
        root_op = root_ops[0]

        parser = dispatch_parser.ContractionOpInterfaceParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = rocm_constraint_generators.ROCmContractionTileAndFuseConstraintGenerator(
            op_info
        )

        assert gen.op_info.dims.batch == []
        assert gen.op_info.dims.m == [0]
        assert gen.op_info.dims.n == [1]
        assert gen.op_info.dims.k == [2]

        assert gen.op_info.matmul_size.M == [30]
        assert gen.op_info.matmul_size.N == [30]
        assert gen.op_info.matmul_size.K == [30]

        assert gen.op_info.lhs_type.shape == [30, 30]
        assert gen.op_info.rhs_type.shape == [30, 30]
        assert gen.op_info.res_type.shape == [30, 30]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                num_subgroups=4,
            )
        )

        assert (
            len(solutions) > 0
        ), "No solutions generated for small unaligned matmul case."

        for solution in solutions:
            assert len(solution) == 1
            config = solution[0]
            assert isinstance(config, common.TuningConfiguration)

            assert config.name == "compilation_info"
            assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)

            lowering_config = config.configuration.lowering_config
            assert "padding =" in str(lowering_config)
            promote = [int(x) for x in lowering_config.attributes["promote_operands"]]
            assert promote == [0, 1]


def test_adjust_problem_size_for_pipeline(
    tuner_ctx: common.TunerContext,
) -> None:
    matmul_size = common.ContractionSizes(
        M=[32],
        N=[64],
        K=[128],
        B=[2],
    )
    contraction_dims = common.ContractionDimensions(
        m=[1],
        n=[2],
        k=[3],
        batch=[0],
    )
    pipeline_options_space = rocm_dispatch_constraints.PipelineOptionsSearchSpace(
        prefetch_num_stages=[2],
        no_reduce_shared_memory_bank_conflicts=[True, False],
        use_igemm_convolution=[None],
    )

    rocm_solutions.adjust_problem_size_for_pipeline(
        contraction_dims=contraction_dims,
        matmul_size=matmul_size,
        dispatch_kind=common.DispatchKind.contraction,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    )
    assert pipeline_options_space.use_igemm_convolution == [None]
    assert matmul_size.K == [128]
    assert contraction_dims.k == [3]

    conv_size = common.ContractionSizes(
        M=[2, 32, 32],
        N=[256],
        K=[3, 3, 512],
    )
    conv_dims = common.ContractionDimensions(
        m=[0, 1, 2],
        n=[3],
        k=[4, 5, 6],
    )
    rocm_solutions.adjust_problem_size_for_pipeline(
        contraction_dims=conv_dims,
        matmul_size=conv_size,
        dispatch_kind=common.DispatchKind.conv,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
    )
    assert pipeline_options_space.use_igemm_convolution == [None]
    assert conv_size.K == [3, 3, 512]
    assert conv_dims.k == [4, 5, 6]

    rocm_solutions.adjust_problem_size_for_pipeline(
        contraction_dims=conv_dims,
        matmul_size=conv_size,
        dispatch_kind=common.DispatchKind.conv,
        pipeline_options_search_space=pipeline_options_space,
        codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
    )
    assert pipeline_options_space.use_igemm_convolution == [True]
    assert conv_size.K == [4608]
    assert conv_dims.k == [4]


def test_adjust_problem_size_for_pipeline_with_igemm_details(
    tuner_ctx: common.TunerContext,
) -> None:
    """Test adjust_problem_size_for_pipeline with IGEMM details from the binding."""
    context = tuner_ctx.mlir_ctx

    with ir.Location.unknown(context):
        module = ir.Module.create()
        build_func_with_conv2d_nhwc_hwcf(
            module=module,
            input_shape=(2, 32, 32, 128),
            kernel_shape=(3, 3, 128, 256),
            output_shape=(2, 30, 30, 256),
            input_type=tuner_ctx.type.f16,
            kernel_type=tuner_ctx.type.f16,
            output_type=tuner_ctx.type.f32,
        )

        root_op_list = iree_codegen.get_tuner_root_ops(module)
        assert len(root_op_list) == 1
        root_op = root_op_list[0]

        parser = dispatch_parser.ConvolutionOpInterfaceParser(root_op, tuner_ctx)
        conv_op_info = parser.get_op_info()
        assert isinstance(conv_op_info, dispatch_parser.ConvolutionOpInfo)
        assert (
            conv_op_info.igemm_details is not None
        ), "IGEMM details should be available for NHWC conv"

        assert conv_op_info.dims.m == [0, 1, 2]
        assert conv_op_info.dims.n == [3]
        assert conv_op_info.dims.k == [4, 5, 6]
        assert conv_op_info.dims.batch == []

        assert conv_op_info.matmul_size.M == [2, 30, 30]
        assert conv_op_info.matmul_size.N == [256]
        assert conv_op_info.matmul_size.K == [3, 3, 128]
        assert conv_op_info.matmul_size.B == []

        conv_dims = common.ContractionDimensions(
            m=list(conv_op_info.dims.m),
            n=list(conv_op_info.dims.n),
            k=list(conv_op_info.dims.k),
            batch=list(conv_op_info.dims.batch),
        )
        conv_size = common.ContractionSizes(
            M=list(conv_op_info.matmul_size.M),
            N=list(conv_op_info.matmul_size.N),
            K=list(conv_op_info.matmul_size.K),
            B=list(conv_op_info.matmul_size.B),
        )

        pipeline_options_space = rocm_dispatch_constraints.PipelineOptionsSearchSpace(
            use_igemm_convolution=[None],
        )

        rocm_solutions.adjust_problem_size_for_pipeline(
            contraction_dims=conv_dims,
            matmul_size=conv_size,
            dispatch_kind=common.DispatchKind.conv,
            pipeline_options_search_space=pipeline_options_space,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
            igemm_details=conv_op_info.igemm_details,
        )

        # Verify that use_igemm_convolution is set.
        assert pipeline_options_space.use_igemm_convolution == [True]

        # Get expected IGEMM dimensions and bounds.
        igemm_maps = [
            map_attr.value
            for map_attr in conv_op_info.igemm_details.igemm_contraction_maps
        ]
        igemm_contraction_dims = linalg.infer_contraction_dimensions_from_maps(
            igemm_maps
        )
        igemm_bounds = list(conv_op_info.igemm_details.igemm_loop_bounds)

        # Verify that dimensions are updated to match IGEMM structure.
        assert conv_dims.m == list(igemm_contraction_dims.m)
        assert conv_dims.n == list(igemm_contraction_dims.n)
        assert conv_dims.k == list(igemm_contraction_dims.k)
        assert conv_dims.batch == list(igemm_contraction_dims.batch)

        # Verify sizes correspond to IGEMM loop bounds.
        expected_m_sizes = [igemm_bounds[i] for i in igemm_contraction_dims.m]
        expected_n_sizes = [igemm_bounds[i] for i in igemm_contraction_dims.n]
        expected_k_sizes = [igemm_bounds[i] for i in igemm_contraction_dims.k]

        assert conv_size.M == expected_m_sizes
        assert conv_size.N == expected_n_sizes
        assert conv_size.K == expected_k_sizes

        # Verify that K is flattened (3*3*128 = 1152).
        assert conv_size.K == [1152]


def test_ContractionZ3Constants_to_meta() -> None:
    matmul_size = common.ContractionSizes(M=[1, 2], N=[1], K=[1], B=[1])
    orig = constraint_generator.ContractionZ3Constants.from_sizes(matmul_size)
    meta = orig.to_meta()
    expected = {
        "m_vals": ["m0", "m1"],
        "n_vals": ["n0"],
        "k_vals": ["k0"],
        "subgroup_m_vals": ["subgroup_m0", "subgroup_m1"],
        "subgroup_n_vals": ["subgroup_n0"],
        "subgroup_size": "subgroup_size",
        "intrinsic_mn": "intrinsic_mn",
        "intrinsic_k": "intrinsic_k",
        "wg_x": "wg_x",
        "wg_y": "wg_y",
        "wg_z": "wg_z",
        "sg_m_cnt": "sg_m_cnt",
        "sg_n_cnt": "sg_n_cnt",
    }
    assert meta == expected


def test_ContractionZ3Constants_from_meta_dict() -> None:
    meta: dict[str, str | list[str]] = {
        "m_vals": ["m0", "m1"],
        "n_vals": ["n0"],
        "k_vals": ["k0"],
        "subgroup_m_vals": ["subgroup_m0", "subgroup_m1"],
        "subgroup_n_vals": ["subgroup_n0"],
        "subgroup_size": "subgroup_size",
        "intrinsic_mn": "intrinsic_mn",
        "intrinsic_k": "intrinsic_k",
        "wg_x": "wg_x",
        "wg_y": "wg_y",
        "wg_z": "wg_z",
        "sg_m_cnt": "sg_m_cnt",
        "sg_n_cnt": "sg_n_cnt",
    }

    ctx = z3.Context()
    recon = constraint_generator.ContractionZ3Constants.from_meta(meta=meta, ctx=ctx)

    assert recon.m_vals == [z3.Int(f"m{i}", ctx) for i in range(2)]
    assert recon.n_vals == [z3.Int(f"n0", ctx)]
    assert recon.k_vals == [z3.Int(f"k0", ctx)]
    assert recon.subgroup_m_vals == [z3.Int(f"subgroup_m{i}", ctx) for i in range(2)]
    assert recon.subgroup_n_vals == [z3.Int(f"subgroup_n0", ctx)]

    assert recon.subgroup_size == z3.Int("subgroup_size", ctx)
    assert recon.intrinsic_mn == z3.Int("intrinsic_mn", ctx)
    assert recon.intrinsic_k == z3.Int("intrinsic_k", ctx)
    assert recon.wg_x == z3.Int("wg_x", ctx)
    assert recon.wg_y == z3.Int("wg_y", ctx)
    assert recon.wg_z == z3.Int("wg_z", ctx)
    assert recon.sg_m_cnt == z3.Int("sg_m_cnt", ctx)
    assert recon.sg_n_cnt == z3.Int("sg_n_cnt", ctx)


def test_Z3Constants_meta_roundtrip() -> None:
    matmul_size = common.ContractionSizes(M=[1], N=[1], K=[1], B=[1])
    orig = constraint_generator.ContractionZ3Constants.from_sizes(matmul_size)
    meta = orig.to_meta()

    ctx = z3.Context()
    recon = constraint_generator.ContractionZ3Constants.from_meta(meta=meta, ctx=ctx)

    for f in constraint_generator.fields(orig):
        orig_field = getattr(orig, f.name)
        recon_field = getattr(recon, f.name)

        if isinstance(orig_field, list):
            assert [v.decl().name() for v in orig_field] == [
                v.decl().name() for v in recon_field
            ]
        else:
            assert orig_field.decl().name() == recon_field.decl().name()


def test_get_z3_solutions() -> None:
    matmul_size = common.ContractionSizes(M=[1], N=[1], K=[1], B=[1])
    z3_constants = constraint_generator.ContractionZ3Constants.from_sizes(matmul_size)

    solver = z3.Solver()
    for v in z3_constants.symbols:
        if v is not z3_constants.wg_x:
            solver.add(v == 0)
    solver.add(z3.Or(z3_constants.wg_x == 0, z3_constants.wg_x == 1))

    z3_constraint_set = constraint_generator.ConstraintSet(
        solver=solver, z3_constants=z3_constants
    )

    results = list(constraint_generator.get_z3_solutions(z3_constraint_set))
    pairs = {(res.wg_x, res.wg_y) for res in results}

    assert pairs == {(0, 0), (1, 0)}
    assert len(results) == 2

    solver.add(z3_constants.wg_y == 1)
    z3_constraint_set = constraint_generator.ConstraintSet(
        solver=solver, z3_constants=z3_constants
    )
    results = list(constraint_generator.get_z3_solutions(z3_constraint_set))
    assert results == []
