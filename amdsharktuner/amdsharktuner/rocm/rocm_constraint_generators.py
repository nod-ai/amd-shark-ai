# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Iterator

from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from .. import common, constraint_generator, dispatch_parser
from . import rocm_parsers, rocm_solutions


class ROCmContractionVectorDistributeConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """
    ROCm Constraint generator for contraction operations using VectorDistribute pipeline.

    Generates tuning configurations for matrix multiplication and related contraction
    operations using the LLVMGPUVectorDistribute lowering pipeline.

    Attributes:
        op_info: ContractionOpInfo containing all contraction operation metadata.
    """

    def __init__(self, op_info: dispatch_parser.ContractionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
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
            **pipeline_constraint_options,
        )


class ROCmConvolutionVectorDistributeConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """
    ROCm Constraint generator for convolution operations using VectorDistribute pipeline.

    Generates tuning configurations for convolution operations using the
    LLVMGPUVectorDistribute lowering pipeline. Supports IGEMM-based convolutions.

    Attributes:
        op_info: ROCmConvolutionOpInfo containing all convolution operation metadata.
    """

    def __init__(self, op_info: rocm_parsers.ROCmConvolutionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        # TODO(Bangtian): Simplify the function signature to accept op_info directly instead of
        # unpacking all individual fields.
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
            igemm_details=self.op_info.igemm_details,
            conv_to_igemm_info=self.op_info.conv_to_igemm_info,
            **pipeline_constraint_options,
        )


class ROCmContractionTileAndFuseConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """
    ROCm Constraint generator for contraction operations using TileAndFuse pipeline.

    Generates tuning configurations for matrix multiplication and related contraction
    operations using the LLVMGPUTileAndFuse lowering pipeline.

    Attributes:
        op_info: ContractionOpInfo containing all contraction operation metadata.
    """

    def __init__(self, op_info: dispatch_parser.ContractionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
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
            **pipeline_constraint_options,
        )


class ROCmConvolutionTileAndFuseConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """
    ROCm Constraint generator for convolution operations using TileAndFuse pipeline.

    Generates tuning configurations for convolution operations using the
    LLVMGPUTileAndFuse lowering pipeline. Supports IGEMM-based convolutions.

    Attributes:
        op_info: ROCmConvolutionOpInfo containing all convolution operation metadata.
    """

    def __init__(self, op_info: rocm_parsers.ROCmConvolutionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
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
            **pipeline_constraint_options,
        )


class ROCmAttentionVectorDistributeConstraintGenerator(
    constraint_generator.ConstraintGenerator
):
    """
    ROCm Constraint generator for the IREE LinalgExt AttentionOp.

    Generates tuning configurations for attention operations.

    Attributes:
        op_info: AttentionOpInfo containing all attention operation metadata.
    """

    def __init__(self, op_info: dispatch_parser.AttentionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        return rocm_solutions.generate_attention_solutions(
            tuner_ctx=tuner_context,
            gpu_target_info=gpu_target_info,
            op_info=self.op_info,
            dispatch_kind=common.DispatchKind.attention,
            codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute,
            **pipeline_constraint_options,
        )
