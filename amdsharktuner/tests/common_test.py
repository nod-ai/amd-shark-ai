# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest common_test.py
"""

import pytest
from dataclasses import dataclass
from types import SimpleNamespace

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import _builtin_ops_gen, func, iree_codegen, iree_gpu, linalg, transform  # type: ignore

from amdsharktuner import common, dispatch_parser
from amdsharktuner.test_utils import tuner_ctx


def test_get_shaped_type_element_bitwidth(tuner_ctx: common.TunerContext) -> None:
    assert common.ShapedType([1024, 2048], tuner_ctx.type.i8).bitwidth == 8
    assert common.ShapedType([2048], tuner_ctx.type.i32).bitwidth == 32
    assert common.ShapedType([2048, 512, 384], tuner_ctx.type.f8E4M3FNUZ).bitwidth == 8
    assert common.ShapedType([1, 1], tuner_ctx.type.f16).bitwidth == 16


def test_get_shaped_type_to_str(tuner_ctx: common.TunerContext) -> None:
    assert str(common.ShapedType([1024, 2048], tuner_ctx.type.i8)) == "1024x2048xi8"
    assert str(common.ShapedType([1024], tuner_ctx.type.f32)) == "1024xf32"
    assert str(common.ShapedType([1, 2, 3], tuner_ctx.type.f16)) == "1x2x3xf16"
    assert str(common.ShapedType([-1, 2, 3], tuner_ctx.type.f16)) == "?x2x3xf16"


def test_gpu_pipeline_options(tuner_ctx: common.TunerContext) -> None:
    options = iree_gpu.PipelineOptionsAttr.get()
    assert str(options) == "#iree_gpu.pipeline_options<>"

    options = iree_gpu.PipelineOptionsAttr.get(prefetch_num_stages=2)
    assert str(options) == "#iree_gpu.pipeline_options<prefetch_num_stages = 2>"

    options = iree_gpu.PipelineOptionsAttr.get(
        prefetch_num_stages=2, no_reduce_shared_memory_bank_conflicts=False
    )
    assert (
        str(options)
        == "#iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false>"
    )

    options = iree_gpu.PipelineOptionsAttr.get(
        reorder_workgroups_strategy=iree_gpu.ReorderWorkgroupsStrategyAttr.get(
            iree_gpu.ReorderWorkgroupsStrategy.Transpose
        )
    )
    assert (
        str(options)
        == "#iree_gpu.pipeline_options<reorder_workgroups_strategy = <Transpose>>"
    )


def test_get_map_result_dim_positions(tuner_ctx: common.TunerContext) -> None:
    dim0 = ir.AffineDimExpr.get(0)
    dim1 = ir.AffineDimExpr.get(1)
    dim2 = ir.AffineDimExpr.get(2)

    # Valid projected permutation: (d0, d1, d2) -> (d0, d2).
    valid_map = ir.AffineMap.get(3, 0, [dim0, dim2])
    result = common.get_map_result_dim_positions(valid_map)
    assert result == [0, 2], f"Expected [0, 2], got {result}"

    # Not a projected permutation: (d0, d1, d2) -> (d0 + d1).
    sum_expr = dim0 + dim1
    invalid_map = ir.AffineMap.get(3, 0, [sum_expr])
    result = common.get_map_result_dim_positions(invalid_map)
    assert result is None, "Expected None for non-projected permutation"


def test_get_pipeline_config(tuner_ctx: common.TunerContext) -> None:
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
    config_dict = common.get_translation_info_config(pipeline_options, 2)
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
    config_dict = common.get_translation_info_config(pipeline_options, 4)
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
    assert common.get_compatible_mma_intrinsics(lhs, rhs, res, all_intrinsics) == [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16,
    ]

    lhs = common.ShapedType([2048, 1280], tuner_ctx.type.i8)
    rhs = common.ShapedType([1280, 1280], tuner_ctx.type.i8)
    res = common.ShapedType([2048, 1280], tuner_ctx.type.i32)
    assert common.get_compatible_mma_intrinsics(lhs, rhs, res, all_intrinsics) == [
        iree_gpu.MMAIntrinsic.MFMA_I32_16x16x32_I8,
        iree_gpu.MMAIntrinsic.MFMA_I32_32x32x16_I8,
    ]

    lhs = common.ShapedType([64, 968, 640], tuner_ctx.type.f32)
    rhs = common.ShapedType([64, 640, 320], tuner_ctx.type.f32)
    res = common.ShapedType([64, 968, 320], tuner_ctx.type.f32)
    assert common.get_compatible_mma_intrinsics(lhs, rhs, res, all_intrinsics) == []

    intrinsics_with_virtual = [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_32x32x16_F8E4M3FNUZ,
    ]

    lhs = common.ShapedType([32, 16], tuner_ctx.type.f8E4M3FNUZ)
    rhs = common.ShapedType([16, 32], tuner_ctx.type.f8E4M3FNUZ)
    res = common.ShapedType([32, 32], tuner_ctx.type.f32)
    assert (
        common.get_compatible_mma_intrinsics(lhs, rhs, res, intrinsics_with_virtual)
        == []
    )

    assert common.get_compatible_mma_intrinsics(
        lhs, rhs, res, intrinsics_with_virtual, allow_virtual_mma=True
    ) == [iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_32x32x16_F8E4M3FNUZ]


def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
    lowering_config = common.get_lowering_config(
        tuner_ctx=tuner_ctx,
        workgroup=[4, 8, 0],
        reduction=[0, 0, 16],
        subgroup_basis=[[1, 1, 1], [0, 1, 2]],
    )

    assert (
        str(lowering_config)
        == "#iree_gpu.lowering_config<{reduction = [0, 0, 16], subgroup_basis = [[1, 1, 1], [0, 1, 2]], workgroup = [4, 8, 0]}>"
    )

    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
    )
    pipeline_options = iree_gpu.PipelineOptionsAttr.get()
    config_dict = common.get_translation_info_config(pipeline_options, 2)
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [16, 16, 1], 32, config_dict
    )
    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )

    assert compilation_info.lowering_config.mma_kind is None
    assert compilation_info.lowering_config.subgroup_basis == ([1, 1, 1], [0, 1, 2])


def test_combine_tuning_specs(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    first_module_str = """
        module @inner_module_a
            attributes { transform.with_named_sequence } {
        }
    """

    second_module_str = """
        module @inner_module_b
            attributes { transform.with_named_sequence } {
        }
    """

    first_ir_module = ir.Module.parse(first_module_str, context)
    second_ir_module = ir.Module.parse(second_module_str, context)

    module = common.combine_tuning_specs(tuner_ctx, [first_ir_module, second_ir_module])
    assert module
    assert "transform.with_named_sequence" in module.operation.attributes

    inner_ops = list(module.body.operations)
    assert all(
        isinstance(op, _builtin_ops_gen.ModuleOp) for op in inner_ops
    ), "Not all ops are builtin.module"
    assert len(inner_ops) == 2, f"Expected 2 inner modules, got {len(inner_ops)}"
    assert (
        inner_ops[0].sym_name.value == "inner_module_a"
    ), f"Expected 'inner_module_a', got '{inner_ops[0].sym_name.value}'"
    assert (
        inner_ops[1].sym_name.value == "inner_module_b"
    ), f"Expected 'inner_module_b', got '{inner_ops[1].sym_name.value}'"


def test_link_tuning_specs(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module @inner_module_a
            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg : !transform.any_op
            }

            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
                transform.yield
            }

            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
                    : (!transform.any_op) -> (!transform.any_op)
                transform.yield %res : !transform.any_op
            }
        }
    """

    ir_module = ir.Module.parse(module_str, context)
    linked_module = common.link_tuning_specs(tuner_ctx, [ir_module])
    assert (
        linked_module is ir_module
    ), "Expected single input module to be returned without modification"

    first_ir_module = ir.Module.parse(module_str, context)
    second_ir_module = ir.Module.parse(module_str, context)
    second_ir_module.operation.attributes["sym_name"] = ir.StringAttr.get(
        "inner_module_b"
    )
    linked_module = common.link_tuning_specs(
        tuner_ctx, [first_ir_module, second_ir_module]
    )
    assert linked_module

    assert "transform.with_named_sequence" in linked_module.operation.attributes
    assert (
        "iree_codegen.tuning_spec_with_default_entrypoint"
        in linked_module.operation.attributes
    )

    inner_ops = list(linked_module.body.operations)
    # Check that inner modules have been merged into the top-level module and no inner modules remain.
    assert all(
        not isinstance(op, _builtin_ops_gen.ModuleOp) for op in inner_ops
    ), "Unexpected inner builtin.module ops found"

    named_sequences = []
    kernel_config_op = None
    for op in linked_module.body.operations:
        if isinstance(op, transform.NamedSequenceOp):
            sym_name_attr = op.sym_name
            assert sym_name_attr is not None
            named_sequences.append(sym_name_attr.value)
            if sym_name_attr.value == "__kernel_config":
                kernel_config_op = op

    assert kernel_config_op is not None, "Missing @__kernel_config"


def test_link_tuning_specs_raises_error(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module @inner_module_a
            attributes { transform.with_named_sequence } {
        }
    """

    module = ir.Module.parse(module_str, context)
    module.operation.attributes[
        "iree_codegen.tuning_spec_with_default_entrypoint"
    ] = ir.UnitAttr.get()
    with pytest.raises(RuntimeError) as exc_info:
        common.link_tuning_specs(tuner_ctx, [module, module])
        # iree-opt should fail due to missing named sequence @__kernel_config entrypoint required
        # by the `iree_codegen.tuning_spec_with_default_entrypoint` attribute.
        assert "iree-opt failed" in str(exc_info.value)


def test_get_matcher_names_from_td_spec(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module attributes { transform.with_named_sequence } {
            transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}) {
                transform.yield
            }

            transform.named_sequence @match_foo(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @match_bar(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @__kernel_config(%arg0: !transform.any_op ) -> !transform.any_op
                attributes { iree_codegen.tuning_spec_entrypoint } {
                %0 = transform.foreach_match in %arg0
                    @match_foo -> @apply_op_config,
                    @match_bar -> @apply_op_config : (!transform.any_op) -> !transform.any_op
                transform.yield %0 : !transform.any_op
            }
        }
    """

    module = ir.Module.parse(module_str, context)
    matcher_names = common.get_matcher_names_from_td_spec(module)

    assert matcher_names == {"match_foo", "match_bar"}

    module_str = """
        module attributes { transform.with_named_sequence } {
            transform.named_sequence @__kernel_config(%arg0: !transform.any_op) -> !transform.any_op
                attributes { iree_codegen.tuning_spec_entrypoint } {
                transform.yield %arg0 : !transform.any_op
            }
        }
    """
    module = ir.Module.parse(module_str, context)
    matcher_names = common.get_matcher_names_from_td_spec(module)
    assert matcher_names == set()


def test_get_matcher_overlap_info(tuner_ctx: common.TunerContext) -> None:
    starter = {"match_a", "match_b", "match_c"}
    current = {"match_b", "match_d"}

    overlapping, unique = common.get_matcher_overlap_info(starter, current)

    assert overlapping == {"match_b"}
    assert unique == {"match_a", "match_c"}

    starter = {"match_x", "match_y"}
    current = {"match_a", "match_b"}
    overlapping, unique = common.get_matcher_overlap_info(starter, current)
    assert overlapping == set()
    assert unique == {"match_x", "match_y"}

    starter = {"match_a", "match_b"}
    current = {"match_a", "match_b", "match_c"}
    overlapping, unique = common.get_matcher_overlap_info(starter, current)
    assert overlapping == {"match_a", "match_b"}
    assert unique == set()


def test_determine_td_specs_to_link(
    tuner_ctx: common.TunerContext, caplog: pytest.LogCaptureFixture
) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        module attributes { transform.with_named_sequence } {
            transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}) {
                transform.yield
            }

            transform.named_sequence @match_foo(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @match_bar(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @__kernel_config(%arg0: !transform.any_op ) -> !transform.any_op
                attributes { iree_codegen.tuning_spec_entrypoint } {
                %0 = transform.foreach_match in %arg0
                    @match_foo -> @apply_op_config,
                    @match_bar -> @apply_op_config : (!transform.any_op) -> !transform.any_op
                transform.yield %0 : !transform.any_op
            }
        }
    """
    starter_td_spec = ir.Module.parse(module_str, context)
    current_td_spec = ir.Module.parse(module_str, context)

    td_specs_to_link = common.determine_td_specs_to_link(
        [current_td_spec, starter_td_spec],
        log_duplicates=True,
    )

    assert td_specs_to_link == [current_td_spec]
    assert "match_foo" in caplog.text
    assert "match_bar" in caplog.text
    assert "already been tuned in the starter" in caplog.text

    caplog.clear()
    module_str = """
        module attributes { transform.with_named_sequence } {
            transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}) {
                transform.yield
            }

            transform.named_sequence @match_baz(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
                transform.yield %arg0 : !transform.any_op
            }

            transform.named_sequence @__kernel_config(%arg0: !transform.any_op ) -> !transform.any_op
                attributes { iree_codegen.tuning_spec_entrypoint } {
                %0 = transform.foreach_match in %arg0
                    @match_baz -> @apply_op_config : (!transform.any_op) -> !transform.any_op
                transform.yield %0 : !transform.any_op
            }
        }
    """
    current_td_spec = ir.Module.parse(module_str, context)
    td_specs_to_link = common.determine_td_specs_to_link(
        [starter_td_spec, current_td_spec],
        log_duplicates=True,
    )

    assert td_specs_to_link == [starter_td_spec, current_td_spec]
    assert "match_baz" not in caplog.text
    assert "already been tuned" not in caplog.text


def test_time_budget() -> None:
    time_budget = common.TimeBudget.for_minutes(-5)
    assert time_budget == None
    time_budget = common.TimeBudget.for_minutes(0)
    assert time_budget == None

    base = 1.234
    time_budget = common.TimeBudget.for_minutes(
        minutes=0.1, now=base
    )  # Set a 6s timer from timestamp `base`.
    assert time_budget != None
    assert time_budget.remaining(base) > 1
    assert time_budget.expired(base) == False

    assert time_budget.remaining(base + 3) > 2
    assert time_budget.expired(base + 3) == False
    assert time_budget.remaining(base + 6) == 0
    assert time_budget.expired(base + 6) == True


def test_get_knob() -> None:
    @dataclass
    class TestKnob(common.KnobAssignment):
        tile_m: int = 64
        wg_x: int = 32
        Tag: bool = True

    test_knob = TestKnob()
    assert test_knob.get_knobs() == {"tile_m": 64, "wg_x": 32, "Tag": True}


def test_get_target_info(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
    hal.executable private @main_dispatch_0 {
        hal.executable.variant public @rocm_hsaco_fb
            target(<"rocm", "rocm-hsaco-fb",
                {
                abi = "hip",
                iree_codegen.target_info = #iree_gpu.target<
                    arch = "gfx942",
                    features = "",
                    wgp = <
                    compute = fp64,
                    storage = b64,
                    subgroup = none,
                    dot = none,
                    mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>],
                    subgroup_size_choices = [32, 64],
                    max_workgroup_sizes = [256, 512, 1024],
                    max_thread_count_per_workgroup = 1024,
                    max_workgroup_memory_bytes = 65536,
                    max_workgroup_counts = [256, 512, 1024],
                    simds_per_wgp = 4
                    >,
                    chip = <wgp_count = 304, sku = "mi300x">
                >
                }>
            ) {
        }
    }
    """

    target_info = common.get_target_info(ir.Module.parse(module_str, context))
    assert target_info

    assert target_info.arch == "gfx942"
    assert target_info.workgroup_count == 304
    assert target_info.simds_per_workgroup == 4
    assert target_info.subgroup_size_choices == [32, 64]
    assert target_info.max_thread_count_per_workgroup == 1024
    assert target_info.max_workgroup_memory_bytes == 65536
    assert target_info.max_workgroup_sizes == [256, 512, 1024]
    assert target_info.mma_intrinsics == [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x4_F32,
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_16x16x32_F16,
    ]


def test_compute_next_aligned_bound() -> None:
    # Already aligned: no padding.
    assert common.compute_next_aligned_bound(128, 128) == 128
    assert common.compute_next_aligned_bound(64, 32) == 64

    # Not aligned: padding needed.
    assert common.compute_next_aligned_bound(100, 128) == 128
    assert common.compute_next_aligned_bound(50, 32) == 64
    assert common.compute_next_aligned_bound(200, 128) == 256


def test_get_dim_bounds() -> None:
    # Padding not expensive: apply padding.
    assert common.get_dim_bounds([200, 300], False) == [256, 384]
    assert common.get_dim_bounds([128, 256], False) == [128, 256]
    assert common.get_dim_bounds([200, 50, 20], False) == [256, 64, 20]

    # Padding expensive: no padding applied.
    assert common.get_dim_bounds([100, 200], True) == [100, 200]


def test_calculate_padded_dimensions(
    tuner_ctx: common.TunerContext,
) -> None:
    with ir.Location.unknown(tuner_ctx.mlir_ctx):
        dim0 = ir.AffineExpr.get_dim(0)
        dim1 = ir.AffineExpr.get_dim(1)
        dim2 = ir.AffineExpr.get_dim(2)

        lhs_map = ir.AffineMap.get(3, 0, [dim0, dim2])
        rhs_map = ir.AffineMap.get(3, 0, [dim2, dim1])
        res_map = ir.AffineMap.get(3, 0, [dim0, dim1])

        contraction_dims = common.ContractionDimensions(m=[0], n=[1], k=[2], batch=[])

        # Test with non-transposed LHS: (m, k) x (k, n) -> (m, n).
        # LHS map: (d0, d1, d2) -> (d0, d2)  # M=d0, K=d2 (non-transposed).
        (M_padded, N_padded, padding_applied,) = common.calculate_padded_dimensions(
            M=[200],
            N=[300],
            contraction_dims=contraction_dims,
            contraction_maps=[lhs_map, rhs_map, res_map],
        )
        assert M_padded == [256], f"Expected M padded to 256, got {M_padded}"
        assert N_padded == [384], f"Expected N padded to 384, got {N_padded}"
        assert padding_applied == True

        # Test with transposed LHS: (k, m) x (k, n) -> (m, n).
        # LHS map: (d0, d1, d2) -> (d2, d0)  # K=d2, M=d0 (transposed).
        lhs_transposed_map = ir.AffineMap.get(3, 0, [dim2, dim0])

        (M_padded, N_padded, padding_applied,) = common.calculate_padded_dimensions(
            M=[200],
            N=[300],
            contraction_dims=contraction_dims,
            contraction_maps=[lhs_transposed_map, rhs_map, res_map],
        )
        assert M_padded == [200], f"Expected M not padded, got {M_padded}"
        assert N_padded == [300], f"Expected N not padded, got {N_padded}"
        assert padding_applied == False


def test_is_affine_expr_function_of_dim(tuner_ctx: common.TunerContext) -> None:
    with tuner_ctx.mlir_ctx:
        d0 = ir.AffineDimExpr.get(0)
        d1 = ir.AffineDimExpr.get(1)

        assert common.is_affine_expr_function_of_dim(d0, 0)
        assert not common.is_affine_expr_function_of_dim(d0, 1)

        c42 = ir.AffineConstantExpr.get(42)
        assert not common.is_affine_expr_function_of_dim(c42, 0)
        assert not common.is_affine_expr_function_of_dim(c42, 1)

        add_expr = d0 + d1
        assert common.is_affine_expr_function_of_dim(add_expr, 0)
        assert common.is_affine_expr_function_of_dim(add_expr, 1)

        mul_expr = d1 * 2
        assert not common.is_affine_expr_function_of_dim(mul_expr, 0)
        assert common.is_affine_expr_function_of_dim(mul_expr, 1)

        complex_expr = (d0 + d1) * 2
        assert common.is_affine_expr_function_of_dim(complex_expr, 0)
        assert common.is_affine_expr_function_of_dim(complex_expr, 1)


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
    result = common.get_padding_conv_sizes(
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
    result = common.get_padding_conv_sizes(
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
    result = common.get_padding_conv_sizes(
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
    result = common.get_padding_conv_sizes(
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
    result = common.get_padding_conv_sizes(
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
    result = common.get_padding_conv_sizes(
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
    result = common.get_padding_conv_sizes(
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
    result = common.get_padding_conv_sizes(
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

    result = common.get_padding_conv_sizes(
        bounds,
        padding_sizes,
        igemm_iterator_types,
        conv_to_igemm_info,
    )
    assert result == [4, 64, 64, 64, 0, 0, 0]
