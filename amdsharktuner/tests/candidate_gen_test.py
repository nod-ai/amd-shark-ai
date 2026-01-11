# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Usage: python -m pytest candidate_gen_test.py

Tests for generic dispatch tuner selection (set_dispatch_tuner).
ROCm-specific tuner tests are in rocm/tuners_test.py.
"""

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen  # type: ignore

from amdsharktuner import candidate_gen, common
from amdsharktuner.rocm import rocm_tuners

from amdsharktuner.test_utils import tuner_ctx


# Sample list of ROCm tuners for testing.
SAMPLE_TUNERS = [
    rocm_tuners.RocmContractionVectorDistributeTuner,
    rocm_tuners.RocmAttentionVectorDistributeTuner,
    rocm_tuners.RocmConvolutionTileAndFuseTuner,
]


def test_set_dispatch_tuner_with_matvec(tuner_ctx: common.TunerContext) -> None:
    # Make sure we do not crash on unsupported root ops (matvec).
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%A: tensor<8x224xf32>, %x: tensor<224xf32>) -> tensor<8xf32> {
                %init = tensor.empty() : tensor<8xf32>
                %y = linalg.matvec {root_op}
                    ins(%A, %x : tensor<8x224xf32>, tensor<224xf32>)
                    outs(%init : tensor<8xf32>) -> tensor<8xf32>
                return %y : tensor<8xf32>
            }
        }"""

    ir_module = ir.Module.parse(module_str, context)

    # Should return None since mat-vec has invalid dimensions (M=[]).
    result = candidate_gen.set_dispatch_tuner(ir_module, tuner_ctx, SAMPLE_TUNERS)
    assert result is None


def test_set_dispatch_tuner_with_unsupported_conv(
    tuner_ctx: common.TunerContext,
) -> None:
    # Make sure we do not crash on unsupported conv layouts (nchw_fchw).
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<2x2048x34x34xi8>, %arg1: tensor<2048x2048x3x3xi8>) -> tensor<2x2048x32x32xi32> {
                %cst = arith.constant 0 : i32
                %0 = tensor.empty() : tensor<2x2048x32x32xi32>
                %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<2x2048x32x32xi32>) -> tensor<2x2048x32x32xi32>
                %2 = linalg.conv_2d_nchw_fchw {root_op}
                    ins(%arg0, %arg1 : tensor<2x2048x34x34xi8>, tensor<2048x2048x3x3xi8>)
                    outs(%1 : tensor<2x2048x32x32xi32>) -> tensor<2x2048x32x32xi32>
                return %2 : tensor<2x2048x32x32xi32>
            }
        }"""

    ir_module = ir.Module.parse(module_str, context)

    # Should return None since conv with nchw_fchw layout is not supported.
    result = candidate_gen.set_dispatch_tuner(ir_module, tuner_ctx, SAMPLE_TUNERS)
    assert result is None


def test_set_dispatch_tuner_no_root_op(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
                %0 = linalg.add
                    ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>)
                    outs(%arg0 : tensor<256xf32>) -> tensor<256xf32>
                return %0 : tensor<256xf32>
            }
        }"""

    ir_module = ir.Module.parse(module_str, context)

    # Should return None since no root_op is found.
    result = candidate_gen.set_dispatch_tuner(ir_module, tuner_ctx, SAMPLE_TUNERS)
    assert result is None


def test_set_dispatch_tuner_multiple_root_ops(tuner_ctx: common.TunerContext) -> None:
    context = tuner_ctx.mlir_ctx
    module_str = """
        builtin.module{
            func.func @test(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
                %0 = linalg.add {root_op}
                    ins(%arg0, %arg1 : tensor<256xf32>, tensor<256xf32>)
                    outs(%arg0 : tensor<256xf32>) -> tensor<256xf32>
                %1 = linalg.mul {root_op}
                    ins(%0, %0 : tensor<256xf32>, tensor<256xf32>)
                    outs(%0 : tensor<256xf32>) -> tensor<256xf32>
                return %1 : tensor<256xf32>
            }
        }"""

    ir_module = ir.Module.parse(module_str, context)

    # Should return None since multiple root_ops are found.
    result = candidate_gen.set_dispatch_tuner(ir_module, tuner_ctx, SAMPLE_TUNERS)
    assert result is None


def test_get_dispatch_tuners() -> None:
    Pipeline = iree_codegen.DispatchLoweringPassPipeline

    assert candidate_gen.get_dispatch_tuners(
        "gfx942", Pipeline.LLVMGPUVectorDistribute
    ) == [
        rocm_tuners.RocmContractionVectorDistributeTuner,
        rocm_tuners.RocmConvolutionVectorDistributeTuner,
        rocm_tuners.RocmAttentionVectorDistributeTuner,
    ]

    assert candidate_gen.get_dispatch_tuners("gfx942", Pipeline.LLVMGPUTileAndFuse) == [
        rocm_tuners.RocmContractionTileAndFuseTuner,
        rocm_tuners.RocmConvolutionTileAndFuseTuner,
    ]

    assert (
        candidate_gen.get_dispatch_tuners("sm_80", Pipeline.LLVMGPUVectorDistribute)
        == []
    )
    assert candidate_gen.get_dispatch_tuners("gfx942", Pipeline.LLVMGPUDistribute) == []
