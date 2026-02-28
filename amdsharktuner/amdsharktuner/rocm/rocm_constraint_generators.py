# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Iterator

from iree.compiler.dialects import iree_codegen, iree_gpu  # type: ignore

from .. import common, constraint_generator, dispatch_parser
from . import rocm_common, rocm_parsers, rocm_solutions


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
        # Verify convolution_dims is set (should always be populated by parsers).
        assert (
            self.op_info.convolution_dims is not None
        ), "convolution_dims must be set for convolution operations"

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
            convolution_dims=self.op_info.convolution_dims,
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
    LLVMGPUTileAndFuse lowering pipeline. By default, enumerates candidates from
    BOTH IGEMM and direct convolution strategies when the operation supports both.

    Attributes:
        op_info: ROCmConvolutionOpInfo containing all convolution operation metadata.
    """

    def __init__(self, op_info: rocm_parsers.ROCmConvolutionOpInfo):
        self.op_info = op_info

    def generate_solutions(
        self,
        tuner_context: common.TunerContext,
        gpu_target_info: iree_gpu.TargetInfo,
        conv_strategy: rocm_common.ConvolutionStrategy = rocm_common.ConvolutionStrategy.both,
        **pipeline_constraint_options,
    ) -> Iterator[list[common.TuningConfiguration]]:
        """Generate candidates from all applicable strategies (IGEMM and/or direct conv)."""
        # Verify convolution_dims is set (should always be populated by parsers).
        assert (
            self.op_info.convolution_dims is not None
        ), "convolution_dims must be set for convolution operations"

        # Generate IGEMM candidates.
        if conv_strategy in ("igemm", "both"):
            tuner_context.logger.info(
                "Generating convolution candidates using IGEMM strategy"
            )
            yield from rocm_solutions.generate_generic_contraction_solutions(
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
                convolution_dims=self.op_info.convolution_dims,
                **pipeline_constraint_options,
            )

        # Generate direct convolution candidates if supported.
        if conv_strategy in ("direct", "both"):
            if self._supports_direct_convolution(tuner_context):
                tuner_context.logger.info(
                    "Generating convolution candidates using direct strategy"
                )
                direct_dims, direct_sizes = self._compute_direct_conv_dimensions()
                # Pass filter loop info so solution generator can add them with tile size 1.
                direct_conv_info: rocm_solutions.DirectConvInfo = {
                    "filter_loop_indices": list(
                        self.op_info.convolution_dims.filter_loop
                    ),
                    "filter_loop_sizes": self.op_info.filter_loop_sizes,
                }
                yield from rocm_solutions.generate_generic_contraction_solutions(
                    tuner_ctx=tuner_context,
                    gpu_target_info=gpu_target_info,
                    contraction_dims=direct_dims,
                    matmul_size=direct_sizes,
                    lhs_type=self.op_info.lhs_type,
                    rhs_type=self.op_info.rhs_type,
                    res_type=self.op_info.res_type,
                    dispatch_kind=common.DispatchKind.conv,
                    indexing_maps=self.op_info.indexing_maps,
                    codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
                    igemm_details=None,
                    conv_to_igemm_info=None,
                    direct_conv_info=direct_conv_info,
                    **pipeline_constraint_options,
                )

    def _supports_direct_convolution(self, tuner_context: common.TunerContext) -> bool:
        """Check if this convolution supports direct convolution lowering.

        Direct convolution requirements (matching IREE's setDirectConvolutionLoweringConfig):
        - Unit strides only.
        - Innermost M, N, K dimensions must be static (not dynamic).

        Note: Empty strides list means the MLIR operation has no explicit strides
        attribute, which defaults to unit strides per MLIR/linalg semantics.

        Returns:
            True if direct convolution is supported, False otherwise.
        """
        # Requirement 1: Unit strides.
        # Empty strides list = no strides attribute in MLIR = defaults to unit strides.
        has_unit_strides = not self.op_info.strides or all(
            s == 1 for s in self.op_info.strides
        )

        if not has_unit_strides:
            tuner_context.logger.debug(
                f"Skipping direct conv: non-unit strides {self.op_info.strides}"
            )
            return False

        # Requirement 2: Static innermost M, N, K dimensions.
        # For direct conv: M = batch + output_image, N = output_channel, K = input_channel.
        # Check the innermost (last) dimension of each.
        import math

        innermost_m = (
            self.op_info.output_image_sizes[-1]
            if self.op_info.output_image_sizes
            else (self.op_info.batch_sizes[-1] if self.op_info.batch_sizes else None)
        )
        innermost_n = (
            self.op_info.output_channel_sizes[-1]
            if self.op_info.output_channel_sizes
            else None
        )
        innermost_k = (
            self.op_info.input_channel_sizes[-1]
            if self.op_info.input_channel_sizes
            else None
        )

        # Dynamic dimensions are represented as negative values (typically -1).
        if innermost_m is None or innermost_m < 0:
            tuner_context.logger.debug(
                "Skipping direct conv: dynamic innermost M dimension"
            )
            return False

        if innermost_n is None or innermost_n < 0:
            tuner_context.logger.debug(
                "Skipping direct conv: dynamic innermost N dimension"
            )
            return False

        if innermost_k is None or innermost_k < 0:
            tuner_context.logger.debug(
                "Skipping direct conv: dynamic innermost K dimension"
            )
            return False

        return True

    def _compute_direct_conv_dimensions(
        self,
    ) -> tuple[common.ContractionDimensions, common.ContractionSizes]:
        """Compute direct convolution dimensions without IGEMM transformation.

        Maps convolution loops to M/N/K following IREE's direct conv approach:
        - M = batch + output_image.
        - N = output_channel.
        - K = input_channel only (filter loops are tiled to 1, not included in schedule).

        This matches IREE's setDirectConvolutionLoweringConfig which only uses
        inputChannel dimensions for the MMA schedule (ConfigUtils.cpp:1883-1892).
        Filter loop dimensions are later tiled to 1 in the final config.

        Precondition: self.op_info.convolution_dims is non-None (enforced by caller).
        """
        conv_dims = self.op_info.convolution_dims
        # Note: convolution_dims is guaranteed non-None by assertion in generate_solutions().
        assert conv_dims is not None
        return (
            common.ContractionDimensions(
                batch=list(conv_dims.depth),
                m=list(conv_dims.batch) + list(conv_dims.output_image),
                n=list(conv_dims.output_channel),
                k=list(conv_dims.input_channel),  # Only input channels for schedule.
            ),
            common.ContractionSizes(
                B=self.op_info.depth_sizes,
                M=self.op_info.batch_sizes + self.op_info.output_image_sizes,
                N=self.op_info.output_channel_sizes,
                K=self.op_info.input_channel_sizes,  # Only input channels for schedule.
            ),
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
