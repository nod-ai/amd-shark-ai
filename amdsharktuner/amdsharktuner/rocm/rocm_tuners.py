# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, iree_gpu, linalg  # type: ignore

from .. import common, dispatch_parser, spec_builder, tuner_base
from . import rocm_parsers


def _materialize_compilation_info_config(
    constraints_op: iree_codegen.ConstraintsOp,
    solution: common.SMTKnobAssignments,
) -> common.TuningConfiguration:
    compilation_info = iree_codegen.materialize_compilation_info(
        constraints_op, solution
    )
    return common.TuningConfiguration(
        name="compilation_info",
        configuration=compilation_info,
    )


def _materialize_configuration_attr_config(
    constraints_op: iree_codegen.ConstraintsOp,
    attr_name: str,
    solution: common.SMTKnobAssignments,
) -> common.TuningConfiguration:
    configuration = iree_codegen.materialize_configuration_attr(
        constraints_op, attr_name, solution
    )
    return common.TuningConfiguration(
        name=attr_name,
        configuration=configuration,
    )


class ROCmContractionVectorDistributeTuner(
    tuner_base.DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_contraction_op(root_op):
            return False

        # Check if contraction has valid dimensions.
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        if not contraction_dims:
            logging.debug("No contraction dimensions found for operation")
            return False

        if not contraction_dims.m or not contraction_dims.n or not contraction_dims.k:
            logging.debug(
                f"Contraction operation with dimensions M={list(contraction_dims.m)}, "
                f"N={list(contraction_dims.n)}, K={list(contraction_dims.k)} "
                f"is not supported by the tuner yet"
            )
            return False

        return True

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ContractionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.contraction

    def get_tuning_configurations(
        self,
        constraints_op: iree_codegen.ConstraintsOp,
        solution: common.SMTKnobAssignments,
    ) -> list[common.TuningConfiguration]:
        return [_materialize_compilation_info_config(constraints_op, solution)]


class ROCmContractionTileAndFuseTuner(
    tuner_base.DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_contraction_op(root_op):
            return False

        # Check if contraction has valid dimensions.
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        if not contraction_dims:
            logging.debug("No contraction dimensions found for operation")
            return False

        if not contraction_dims.m or not contraction_dims.n or not contraction_dims.k:
            logging.debug(
                f"Contraction operation with dimensions M={list(contraction_dims.m)}, "
                f"N={list(contraction_dims.n)}, K={list(contraction_dims.k)} "
                f"is not supported by the tuner yet"
            )
            return False

        return True

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ContractionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.contraction

    def get_tuning_configurations(
        self,
        constraints_op: iree_codegen.ConstraintsOp,
        solution: common.SMTKnobAssignments,
    ) -> list[common.TuningConfiguration]:
        return [_materialize_compilation_info_config(constraints_op, solution)]


class ROCmConvolutionVectorDistributeTuner(
    tuner_base.DispatchTuner, rocm_parsers.InnerMNKConvolutionParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_convolution_op(root_op):
            return False
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        if not convolution_dims:
            return False
        # Only allow 'nhwc_hwcf' convs.
        return (
            list(convolution_dims.batch) == [0]
            and list(convolution_dims.output_image) == [1, 2]
            and list(convolution_dims.output_channel) == [3]
            and list(convolution_dims.filter_loop) == [4, 5]
            and list(convolution_dims.input_channel) == [6]
            and list(convolution_dims.depth) == []
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ConvolutionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.conv

    def get_tuning_configurations(
        self,
        constraints_op: iree_codegen.ConstraintsOp,
        solution: common.SMTKnobAssignments,
    ) -> list[common.TuningConfiguration]:
        return [_materialize_compilation_info_config(constraints_op, solution)]


class ROCmConvolutionTileAndFuseTuner(
    tuner_base.DispatchTuner, rocm_parsers.IGEMMConvolutionParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_convolution_op(root_op):
            return False
        convolution_dims = linalg.infer_convolution_dimensions(root_op)
        if not convolution_dims:
            return False
        return True

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ConvolutionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.conv

    def get_tuning_configurations(
        self,
        constraints_op: iree_codegen.ConstraintsOp,
        solution: common.SMTKnobAssignments,
    ) -> list[common.TuningConfiguration]:
        return [_materialize_compilation_info_config(constraints_op, solution)]


class ROCmAttentionVectorDistributeTuner(
    tuner_base.DispatchTuner, dispatch_parser.AttentionOpInterfaceParser
):
    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        return iree_codegen.isa_attention_op(root_op)

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.AttentionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.attention

    def get_tuning_configurations(
        self,
        constraints_op: iree_codegen.ConstraintsOp,
        solution: common.SMTKnobAssignments,
    ) -> list[common.TuningConfiguration]:
        compilation_info = _materialize_compilation_info_config(
            constraints_op, solution
        )
        decomposition_config = _materialize_configuration_attr_config(
            constraints_op, "decomposition_config", solution
        )
        return [
            compilation_info,
            decomposition_config,
        ]


def get_tuners_for_pipeline(
    codegen_pipeline: iree_gpu.LoweringPipeline,
) -> list[type[tuner_base.DispatchTuner]]:
    """Get ROCm tuners for the given codegen pipeline."""
    if codegen_pipeline == iree_gpu.LoweringPipeline.VectorDistribute:
        return [
            ROCmContractionVectorDistributeTuner,
            ROCmConvolutionVectorDistributeTuner,
            ROCmAttentionVectorDistributeTuner,
        ]

    if codegen_pipeline == iree_gpu.LoweringPipeline.TileAndFuse:
        return [
            ROCmContractionTileAndFuseTuner,
            ROCmConvolutionTileAndFuseTuner,  # Handles both IGEMM and direct conv strategies
        ]

    return []
