# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from amdsharktuner import common, dispatch_parser, spec_builder
from amdsharktuner.rocm import rocm_dispatch_constraints

from amdsharktuner.test_utils import tuner_ctx  # noqa: F401


def _build_dummy_compilation_info(
    tuner_ctx: common.TunerContext, num_loops: int, reduction_dim: int
) -> iree_codegen.CompilationInfoAttr:
    workgroup = [0] * num_loops
    partial_reduction = [0] * num_loops
    thread = [0] * num_loops
    workgroup[0] = 4
    partial_reduction[reduction_dim] = 256 * 8
    thread[reduction_dim] = 8
    lane_counts = [1] * num_loops
    lane_counts[reduction_dim] = 64
    subgroup_counts = [1] * num_loops
    subgroup_counts[reduction_dim] = 4
    mapping = list(range(num_loops))
    return (
        rocm_dispatch_constraints.generate_matvec_vector_distribute_compilation_infos(
            tuner_ctx=tuner_ctx,
            workgroup_tile_sizes=workgroup,
            partial_reduction_tile_sizes=partial_reduction,
            thread_tile_sizes=thread,
            lane_basis=[lane_counts, mapping],
            subgroup_basis=[subgroup_counts, mapping],
            workgroup_size=256,
            subgroup_size=64,
        )
    )


def test_matvec_spec_builder_emits_vector_distribute(
    tuner_ctx: common.TunerContext,
) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module {
            func.func @test(%A: tensor<4096x4096xf16>, %x: tensor<4096xf16>) -> tensor<4096xf32> {
                %cst = arith.constant 0.0 : f32
                %init = tensor.empty() : tensor<4096xf32>
                %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4096xf32>) -> tensor<4096xf32>
                %y = linalg.matvec {root_op = #iree_codegen.root_op<set = 0>}
                    ins(%A, %x : tensor<4096x4096xf16>, tensor<4096xf16>)
                    outs(%fill : tensor<4096xf32>) -> tensor<4096xf32>
                return %y : tensor<4096xf32>
            }
        }
    """
    ir_module = ir.Module.parse(module_str, context)
    root_op = iree_codegen.get_tuner_root_ops(ir_module)[0]
    op_info = dispatch_parser.MatvecOpInterfaceParser(root_op, tuner_ctx).get_op_info()

    compilation_info = _build_dummy_compilation_info(
        tuner_ctx, op_info.num_loops, op_info.reduction_dim_index
    )
    config_list = [
        common.TuningConfiguration(
            name="compilation_info", configuration=compilation_info
        )
    ]

    builder = spec_builder.MatvecSpecBuilder(op_info)
    td_spec_module = builder.build_td_spec(tuner_ctx, config_list)

    td_str = str(td_spec_module)
    assert "@__kernel_config" in td_str
    assert "apply_op_config" in td_str
    assert "match_test" in td_str
    assert "VectorDistribute" in td_str
    assert "partial_reduction" in td_str
    assert "lane_basis" in td_str
