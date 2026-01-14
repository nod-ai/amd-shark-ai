# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Optional

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore

from .. import common


# The Key name for the 'amdgpu-waves-per-eu' within the llvm_func_attrs attribute.
WAVES_PER_EU_KEY = "amdgpu-waves-per-eu"


@dataclass
class LLVMGPUVectorDistributeContractionKnobs(common.KnobAssignment):
    # Problem Size.
    M: int
    N: int
    K: int

    # Z3 numeric selections.
    tile_m: int
    tile_n: int
    tile_k: int
    wg_x: int
    wg_y: int
    wg_z: int
    subgroup_m_cnt: int
    subgroup_n_cnt: int
    intrinsic_mn: int
    intrinsic_k: int
    subgroup_m: int
    subgroup_n: int
    subgroup_k: int


@dataclass
class ConvolutionKnobs(common.KnobAssignment):
    pass


@dataclass
class AttentionKnobs(common.KnobAssignment):
    pass


def get_compatible_mma_intrinsics(
    lhs_type: common.ShapedType,
    rhs_type: common.ShapedType,
    res_type: common.ShapedType,
    mma_intrinsics: list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic],
    allow_virtual_mma: bool = False,
) -> list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic]:
    def is_compatible(
        mma: iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic,
    ) -> bool:
        # Filter out virtual intrinsics unless explicitly allowed (for attention ops).
        is_virtual = isinstance(mma, iree_gpu.VirtualMMAIntrinsic)
        if is_virtual and not allow_virtual_mma:
            return False

        mma_attr = (
            iree_gpu.VirtualMMAAttr.get(mma)
            if is_virtual
            else iree_gpu.MMAAttr.get(mma)
        )
        a_type, b_type, c_type = mma_attr.abc_element_types
        return (
            lhs_type.element_type == a_type
            and rhs_type.element_type == b_type
            and res_type.element_type == c_type
        )

    return list(filter(is_compatible, mma_intrinsics))


# Generate a config dictionary used in translation_info attribute.
def get_translation_info_config(
    pipeline_options: iree_gpu.PipelineOptionsAttr, waves_per_eu: int
) -> ir.DictAttr:
    """
    Example IR
    translation_info = #iree_codegen.translation_info<
                    pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64,
                    {gpu_pipeline_options = #iree_gpu.pipeline_options<...>,
                     llvm_func_attrs = {"amdgpu-waves-per-eu" = "3"}
                    }
                >
    """
    waves_per_eu_str = str(waves_per_eu)

    # Create the waves_per_eu dictionary attribute.
    waves_per_eu_dict = ir.DictAttr.get(
        {WAVES_PER_EU_KEY: ir.StringAttr.get(waves_per_eu_str)}
    )

    config_dict = ir.DictAttr.get(
        {
            common.GPU_PIPELINE_OPTIONS_KEY: pipeline_options,
            common.LLVM_FUNC_ATTRS_KEY: waves_per_eu_dict,
        }
    )

    return config_dict


def get_attention_decomposition_config(
    tuner_ctx: common.TunerContext,
    qk_lowering_config: iree_gpu.LoweringConfigAttr,
    pv_lowering_config: iree_gpu.LoweringConfigAttr,
) -> ir.DictAttr:
    """
    Constructs the decomposition config for an attention op, embedding
    separate lowering configs for QK and PV matmuls.
    """

    ctx = tuner_ctx.mlir_ctx
    qk_attrs_dict = {
        "attention_qk_matmul": ir.UnitAttr.get(ctx),
        "lowering_config": qk_lowering_config,
    }
    qk_attr_dict = ir.DictAttr.get(qk_attrs_dict, context=ctx)

    pv_attrs_dict = {
        "attention_pv_matmul": ir.UnitAttr.get(ctx),
        "lowering_config": pv_lowering_config,
    }
    pv_attr_dict = ir.DictAttr.get(pv_attrs_dict, context=ctx)

    decomposition_config_dict = {
        "qk_attrs": qk_attr_dict,
        "pv_attrs": pv_attr_dict,
    }

    return ir.DictAttr.get(decomposition_config_dict, context=ctx)


# Implemented the logic from IREE side:
# https://github.com/iree-org/iree/blob/8ae91ebb0e555e660b8a6898f6071476f7a1f20b/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L382-L467.
def get_padding_conv_sizes(
    bounds: list[int],
    padding_sizes: list[int],
    igemm_loop_iterators: list[str],
    conv_to_igemm_info: common.ConvToIgemmInfo,
) -> Optional[list[int]]:
    """
    Computes padding_conv by mapping padding from IGEMM space to convolution space.

    Args:
        bounds: Loop bounds for each dimension.
        padding_sizes: Padding sizes in IGEMM dimension space (M, N, K).
        igemm_loop_iterators: IGEMM loop iterator type strings ('"reduction"' or '"parallel"').
        conv_to_igemm_info: Convolution to IGEMM transformation info.

    Returns:
        Padding sizes in convolution dimension space, or None if no padding
        is needed along original convolution dimensions.
    """
    # Skip padding convolution for NCHW layout (spatial dimensions are last).
    if conv_to_igemm_info.is_spatial_dim_last:
        return None

    conv_to_igemm_map = conv_to_igemm_info.conv_to_igemm_dim_map
    padded_igemm_dims = set()
    conv_dims = conv_to_igemm_info.conv_dims
    input_channel_dims = set(conv_dims.input_channel)

    padding_conv_sizes = [0] * len(conv_to_igemm_map)

    # For batch-last layout (e.g., CHWN), only pad the batch dimension to avoid
    # introducing pad op as the producer of collapse_shape op which may cause fusion problem.
    if conv_to_igemm_info.is_batch_dim_last:
        last_batch_dim = conv_dims.batch[-1]
        igemm_batch_pos = conv_to_igemm_map[last_batch_dim]

        if (
            padding_sizes[igemm_batch_pos]
            and bounds[igemm_batch_pos] % padding_sizes[igemm_batch_pos] == 0
        ):
            return None

        padding_conv_sizes[last_batch_dim] = padding_sizes[igemm_batch_pos]
        return padding_conv_sizes

    for conv_dim, igemm_pos in conv_to_igemm_map.items():
        if igemm_loop_iterators[igemm_pos] == '"reduction"':
            # Skip filter loop dimensions (reduction dims that aren't input channels).
            # Only pad input channel dims. If we need to pad filter dims, then we
            # would rather just do padding on the IGEMM instead.
            if conv_dim not in input_channel_dims:
                continue

            # Skip conv padding for input channel dims if already divisible by padding size.
            if (
                padding_sizes[igemm_pos]
                and bounds[igemm_pos] % padding_sizes[igemm_pos] == 0
            ):
                padded_igemm_dims.add(igemm_pos)
                continue

            # Multiple input channel dims for a single IGEMMPos is not supported.
            if igemm_pos in padded_igemm_dims:
                return None

            input_channel_size = conv_to_igemm_info.input_channel_dim_to_size.get(
                conv_dim, 0
            )
            is_input_channel_size_small = (
                padding_sizes[igemm_pos] // input_channel_size > 2
            )

            # If the input channel dimension is much smaller than the padding size,
            # skip padding along that dimension while still padding the others.
            if is_input_channel_size_small:
                padding_conv_sizes[conv_dim] = 0
            else:
                padding_conv_sizes[conv_dim] = padding_sizes[igemm_pos]

            padded_igemm_dims.add(igemm_pos)
            continue

        # Multiple padded parallel dims mapping to the same IGEMM dim is not supported.
        if padding_sizes[igemm_pos] and igemm_pos in padded_igemm_dims:
            return None

        padding_conv_sizes[conv_dim] = padding_sizes[igemm_pos]
        padded_igemm_dims.add(igemm_pos)

    # Ensure that all dimensions have been padded.
    if len(padded_igemm_dims) != len(padding_sizes):
        return None

    return padding_conv_sizes
