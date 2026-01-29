# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

# TODO: remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
import amdsharktuner
from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from amdsharktuner import (
    common,
    dispatch_parser,
)
from amdsharktuner.rocm import (
    rocm_constraint_generators,
    rocm_dispatch_constraints,
    rocm_parsers,
    rocm_solutions,
)

from amdsharktuner.test_utils import tuner_ctx
from tests.constraint_generator_test import (
    build_func_with_matmul,
    build_func_with_conv2d_nhwc_hwcf,
    build_func_with_conv2d_nchw_fchw,
    build_func_with_conv2d_nhwc_fhwc,
    build_func_with_conv2d_chwn_chwf,
)


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
            iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
            iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
            iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
        ],
    )


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

        parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
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
    """Test TileAndFuse with small dimensions (< 32) unaligned to intrinsics."""
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

        parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
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

        parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = rocm_constraint_generators.ROCmConvolutionTileAndFuseConstraintGenerator(
            op_info
        )

        assert gen.op_info.lhs_type.shape == [2, 128, 34, 34]
        assert gen.op_info.rhs_type.shape == [256, 128, 3, 3]
        assert gen.op_info.res_type.shape == [2, 256, 32, 32]
        assert gen.op_info.matmul_size.K == [1152]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
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

        parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = rocm_constraint_generators.ROCmConvolutionTileAndFuseConstraintGenerator(
            op_info
        )

        assert gen.op_info.lhs_type.shape == [2, 10, 10, 32]
        assert gen.op_info.rhs_type.shape == [64, 3, 3, 32]
        assert gen.op_info.res_type.shape == [2, 8, 8, 64]
        assert gen.op_info.matmul_size.K == [288]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
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

        parser = rocm_parsers.IGEMMConvolutionParser(root_op, tuner_ctx)
        op_info = parser.get_op_info()
        gen = rocm_constraint_generators.ROCmConvolutionTileAndFuseConstraintGenerator(
            op_info
        )

        assert gen.op_info.lhs_type.shape == [32, 10, 10, 2]
        assert gen.op_info.rhs_type.shape == [32, 3, 3, 64]
        assert gen.op_info.res_type.shape == [64, 8, 8, 2]
        assert gen.op_info.matmul_size.K == [288]

        solutions = list(
            gen.generate_solutions(
                tuner_context=tuner_ctx,
                gpu_target_info=gpu_target_info,
                num_subgroups=4,
            )
        )

        assert len(solutions) > 0, "No solutions generated for CHWN_CHWF conv."
        for solution in solutions:
            assert len(solution) == 1
            config = solution[0]
            assert isinstance(config, common.TuningConfiguration)
            assert isinstance(config.configuration, iree_codegen.CompilationInfoAttr)
