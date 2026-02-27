# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ROCm-specific convolution parsers for IGEMM and INNER_MNK lowering strategies."""

from dataclasses import dataclass
from typing import Optional

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, linalg  # type: ignore

from .. import common, dispatch_parser
from .rocm_common import ConvToIgemmInfo


@dataclass
class ROCmConvolutionOpInfo(dispatch_parser.ConvolutionOpInfo):
    """ROCm-specific convolution operation info with IGEMM transformation details.

    IGEMM (Implicit GEMM) is an IREE convolution lowering strategy used with the
    TileAndFuse pipeline that transforms convolutions into matrix multiplications
    by flattening spatial and channel dimensions. This subclass extends the generic
    ConvolutionOpInfo with IGEMM transformation metadata needed for tuning and
    padding calculations.
    """

    # IGEMM contraction maps, loop bounds, and iterator types from IREE compiler.
    # Used by TileAndFuse pipeline for dimension flattening (e.g., K = FH*FW*IC).
    igemm_details: Optional[iree_codegen.IGEMMGenericConvDetails] = None

    # Mapping between original convolution dimensions and IGEMM dimensions.
    # Used to calculate padding_conv attribute by mapping padding from IGEMM
    # space (M, N, K) back to convolution space (batch, spatial, channels).
    conv_to_igemm_info: Optional[ConvToIgemmInfo] = None

    # Original convolution dimensions.
    # Used for both IGEMM and direct convolution dimension computations.
    # Contains: batch, output_image, output_channel, filter_loop, input_channel, depth.
    # Note: Always populated by parsers, but Optional due to dataclass field ordering.
    convolution_dims: Optional[linalg.ConvolutionDimensions] = None


def build_conv_to_igemm_info(
    convolution_dims: linalg.ConvolutionDimensions,
    input_type: ir.Type,
    input_map: ir.AffineMap,
    igemm_details: iree_codegen.IGEMMGenericConvDetails,
) -> ConvToIgemmInfo:
    """
    Builds ConvToIgemmInfo from convolution dimensions and IGEMM details.

    Corresponds to IREE:
    https://github.com/iree-org/iree/blob/d3440737cc56a4d1b20c72181d9a37f194bd3ce5/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L872-L909
    """
    input_shape = input_type.shape
    conv_to_igemm_info = ConvToIgemmInfo()

    # Map input channel dimensions to their sizes in the input tensor.
    for dim in convolution_dims.input_channel:
        for idx, expr in enumerate(input_map.results):
            if common.is_affine_expr_function_of_dim(expr, dim):
                conv_to_igemm_info.input_channel_dim_to_size[dim] = input_shape[idx]

    # Process output image dimensions to find input image positions.
    input_image_pos = []
    for dim in convolution_dims.output_image:
        for idx, expr in enumerate(input_map.results):
            if common.is_affine_expr_function_of_dim(expr, dim):
                input_image_pos.append(idx)

    # Process batch dimensions to find batch positions.
    batch_pos = []
    for dim in convolution_dims.batch:
        for idx, expr in enumerate(input_map.results):
            if common.is_affine_expr_function_of_dim(expr, dim):
                batch_pos.append(idx)

    input_image_pos = sorted(input_image_pos)
    batch_pos = sorted(batch_pos)

    conv_to_igemm_info.is_batch_dim_last = (
        len(batch_pos) > 0 and batch_pos[-1] == len(input_shape) - 1
    )
    conv_to_igemm_info.is_spatial_dim_last = (
        len(input_image_pos) > 0 and input_image_pos[-1] == len(input_shape) - 1
    )

    conv_to_igemm_info.conv_to_igemm_dim = dict(igemm_details.conv_to_igemm_dim_map)
    return conv_to_igemm_info


class IGEMMConvolutionParser(dispatch_parser.ConvolutionOpInterfaceParser):
    """Convolution parser for IGEMM (Implicit GEMM) with LLVMGPUTileAndFuse pipeline.

    Supports all 2D convolution layouts. Flattens K dimension (filter + input channels).
    """

    def __init__(
        self,
        root_op: ir.Operation,
        tuner_ctx: common.TunerContext,
    ):
        super().__init__(root_op, tuner_ctx)
        info = self._conv_dim_info

        # Get IGEMM details for the TileAndFuse pipeline.
        igemm_details = iree_codegen.get_igemm_generic_conv_details(root_op)
        assert igemm_details, "Failed to get IGEMM details for convolution"

        igemm_maps = [
            map_attr.value for map_attr in igemm_details.igemm_contraction_maps
        ]
        igemm_contraction_dims = linalg.infer_contraction_dimensions_from_maps(
            igemm_maps
        )
        assert (
            igemm_contraction_dims
        ), "Failed to infer contraction dimensions from IGEMM maps"

        bounds = list(igemm_details.igemm_loop_bounds)

        contraction_dims = common.ContractionDimensions(
            batch=list(igemm_contraction_dims.batch),
            m=list(igemm_contraction_dims.m),
            n=list(igemm_contraction_dims.n),
            k=list(igemm_contraction_dims.k),
        )
        matmul_size = common.ContractionSizes(
            B=[bounds[i] for i in contraction_dims.batch],
            M=[bounds[i] for i in contraction_dims.m],
            N=[bounds[i] for i in contraction_dims.n],
            K=[bounds[i] for i in contraction_dims.k],
        )

        conv_to_igemm_info = build_conv_to_igemm_info(
            info.convolution_dims, info.lhs_type, info.indexing_maps[0], igemm_details
        )

        # Build ROCm-specific op info with IGEMM details.
        self._op_info = ROCmConvolutionOpInfo(
            root_op=self.get_root_op(),
            indexing_maps=info.indexing_maps,
            dims=contraction_dims,
            matmul_size=matmul_size,
            lhs_type=common.ShapedType(info.lhs_type.shape, info.lhs_type.element_type),
            rhs_type=common.ShapedType(info.rhs_type.shape, info.rhs_type.element_type),
            res_type=common.ShapedType(info.res_type.shape, info.res_type.element_type),
            batch_sizes=info.batch_sizes,
            output_image_sizes=info.output_image_sizes,
            output_channel_sizes=info.output_channel_sizes,
            filter_loop_sizes=info.filter_loop_sizes,
            input_channel_sizes=info.input_channel_sizes,
            depth_sizes=info.depth_sizes,
            strides=info.strides,
            dilations=info.dilations,
            igemm_details=igemm_details,
            conv_to_igemm_info=conv_to_igemm_info,
            convolution_dims=info.convolution_dims,
        )

    def get_op_info(self) -> ROCmConvolutionOpInfo:
        assert isinstance(
            self._op_info, ROCmConvolutionOpInfo
        ), "Failed to build ROCmConvolutionOpInfo"
        return self._op_info


class InnerMNKConvolutionParser(dispatch_parser.ConvolutionOpInterfaceParser):
    """Convolution parser for INNER_MNK with LLVMGPUVectorDistribute pipeline.

    Supports NHWC_HWCF layout only. Uses conv dimensions directly without flattening.
    """

    def __init__(
        self,
        root_op: ir.Operation,
        tuner_ctx: common.TunerContext,
    ):
        super().__init__(root_op, tuner_ctx)
        info = self._conv_dim_info

        # INNER_MNK: Use the convolution dimensions directly.
        contraction_dims = common.ContractionDimensions(
            batch=info.depth_indices,
            m=info.batch_indices + info.output_image_indices,
            n=info.output_channel_indices,
            k=info.filter_loop_indices + info.input_channel_indices,
        )
        matmul_size = common.ContractionSizes(
            B=info.depth_sizes,
            M=info.batch_sizes + info.output_image_sizes,
            N=info.output_channel_sizes,
            K=info.filter_loop_sizes + info.input_channel_sizes,
        )

        # Build ROCm-specific op info without IGEMM details (not needed for INNER_MNK).
        self._op_info = ROCmConvolutionOpInfo(
            root_op=self.get_root_op(),
            indexing_maps=info.indexing_maps,
            dims=contraction_dims,
            matmul_size=matmul_size,
            lhs_type=common.ShapedType(info.lhs_type.shape, info.lhs_type.element_type),
            rhs_type=common.ShapedType(info.rhs_type.shape, info.rhs_type.element_type),
            res_type=common.ShapedType(info.res_type.shape, info.res_type.element_type),
            batch_sizes=info.batch_sizes,
            output_image_sizes=info.output_image_sizes,
            output_channel_sizes=info.output_channel_sizes,
            filter_loop_sizes=info.filter_loop_sizes,
            input_channel_sizes=info.input_channel_sizes,
            depth_sizes=info.depth_sizes,
            strides=info.strides,
            dilations=info.dilations,
            convolution_dims=info.convolution_dims,
        )

    def get_op_info(self) -> ROCmConvolutionOpInfo:
        assert isinstance(
            self._op_info, ROCmConvolutionOpInfo
        ), "Failed to build ROCmConvolutionOpInfo"
        return self._op_info
