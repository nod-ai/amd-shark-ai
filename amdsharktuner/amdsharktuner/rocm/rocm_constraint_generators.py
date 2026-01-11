# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Iterator

from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from .. import common, constraint_generator, dispatch_parser
from . import rocm_solutions


class RocmContractionVectorDistributeConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """ROCm constraint generator for contractions using VectorDistribute pipeline."""

    def __init__(self, op_info: dispatch_parser.ContractionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return rocm_solutions.generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=self.op_info.dims,
            matmul_size=self.op_info.matmul_size,
            lhs_type=self.op_info.lhs_type,
            rhs_type=self.op_info.rhs_type,
            res_type=self.op_info.res_type,
            dispatch_kind=common.DispatchKind.contraction,
            indexing_maps=self.op_info.indexing_maps,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            **constraint_options,
        )


class RocmContractionTileAndFuseConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """ROCm constraint generator for contractions using TileAndFuse pipeline."""

    def __init__(self, op_info: dispatch_parser.ContractionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return rocm_solutions.generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=self.op_info.dims,
            matmul_size=self.op_info.matmul_size,
            lhs_type=self.op_info.lhs_type,
            rhs_type=self.op_info.rhs_type,
            res_type=self.op_info.res_type,
            dispatch_kind=common.DispatchKind.contraction,
            indexing_maps=self.op_info.indexing_maps,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
            **constraint_options,
        )


class RocmConvolutionVectorDistributeConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """ROCm constraint generator for convolutions using VectorDistribute pipeline."""

    def __init__(self, op_info: dispatch_parser.ConvolutionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return rocm_solutions.generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=self.op_info.dims,
            matmul_size=self.op_info.matmul_size,
            lhs_type=self.op_info.lhs_type,
            rhs_type=self.op_info.rhs_type,
            res_type=self.op_info.res_type,
            dispatch_kind=common.DispatchKind.conv,
            indexing_maps=self.op_info.indexing_maps,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            **constraint_options,
        )


class RocmConvolutionTileAndFuseConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """ROCm constraint generator for convolutions using TileAndFuse pipeline (IGEMM)."""

    def __init__(self, op_info: dispatch_parser.ConvolutionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return rocm_solutions.generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            contraction_dims=self.op_info.dims,
            matmul_size=self.op_info.matmul_size,
            lhs_type=self.op_info.lhs_type,
            rhs_type=self.op_info.rhs_type,
            res_type=self.op_info.res_type,
            dispatch_kind=common.DispatchKind.conv,
            indexing_maps=self.op_info.indexing_maps,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
            igemm_details=self.op_info.igemm_details,
            conv_to_igemm_info=self.op_info.conv_to_igemm_info,
            **constraint_options,
        )


class RocmAttentionVectorDistributeConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """ROCm constraint generator for attention using VectorDistribute pipeline."""

    def __init__(self, op_info: dispatch_parser.AttentionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return rocm_solutions.generate_attention_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            op_info=self.op_info,
            dispatch_kind=common.DispatchKind.attention,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            **constraint_options,
        )
