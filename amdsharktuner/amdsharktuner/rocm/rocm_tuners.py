# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import Optional

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_codegen, linalg  # type: ignore

from .. import common, constraint_generator, dispatch_parser, spec_builder
from ..tuner_base import DispatchTuner
from . import rocm_constraint_generators


class RocmContractionVectorDistributeTuner(
    DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
):
    """ROCm tuner for contraction ops using VectorDistribute pipeline."""

    codegen_pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute

    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_contraction_op(root_op):
            return False

        # Check if contraction has valid dimensions.
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        if not contraction_dims:
            logging.warning("No contraction dimensions found for operation")
            return False

        if not contraction_dims.m or not contraction_dims.n or not contraction_dims.k:
            logging.warning(
                f"Contraction operation with dimensions M={list(contraction_dims.m)}, "
                f"N={list(contraction_dims.n)}, K={list(contraction_dims.k)} "
                f"is not supported by the tuner yet"
            )
            return False

        return True

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return rocm_constraint_generators.RocmContractionVectorDistributeConstraintGenerator(
            self.get_op_info()
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ContractionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.contraction

    def get_knob_assignment(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> Optional[common.KnobAssignment]:
        return config_list[0].knob_assignment


class RocmContractionTileAndFuseTuner(
    DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
):
    """ROCm tuner for contraction ops using TileAndFuse pipeline."""

    codegen_pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse

    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        if not linalg.isa_contraction_op(root_op):
            return False

        # Check if contraction has valid dimensions.
        contraction_dims = linalg.infer_contraction_dimensions(root_op)
        if not contraction_dims:
            logging.warning("No contraction dimensions found for operation")
            return False

        if not contraction_dims.m or not contraction_dims.n or not contraction_dims.k:
            logging.warning(
                f"Contraction operation with dimensions M={list(contraction_dims.m)}, "
                f"N={list(contraction_dims.n)}, K={list(contraction_dims.k)} "
                f"is not supported by the tuner yet"
            )
            return False

        return True

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return rocm_constraint_generators.RocmContractionTileAndFuseConstraintGenerator(
            self.get_op_info()
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.ContractionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.contraction

    def get_knob_assignment(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> Optional[common.KnobAssignment]:
        return config_list[0].knob_assignment


class RocmConvolutionVectorDistributeTuner(
    DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
):
    """ROCm tuner for convolution ops using VectorDistribute pipeline."""

    codegen_pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute

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

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return rocm_constraint_generators.RocmConvolutionVectorDistributeConstraintGenerator(
            self.get_op_info()
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

    def get_knob_assignment(
        self,
        _config_list: list[common.TuningConfiguration],
    ) -> Optional[common.KnobAssignment]:
        return None


class RocmConvolutionTileAndFuseTuner(
    DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
):
    """ROCm tuner for convolution ops using TileAndFuse pipeline (IGEMM)."""

    codegen_pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse

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

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return rocm_constraint_generators.RocmConvolutionTileAndFuseConstraintGenerator(
            self.get_op_info()
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

    def get_knob_assignment(
        self,
        _config_list: list[common.TuningConfiguration],
    ) -> Optional[common.KnobAssignment]:
        return None


class RocmAttentionVectorDistributeTuner(
    DispatchTuner, dispatch_parser.AttentionOpInterfaceParser
):
    """ROCm tuner for attention ops using VectorDistribute pipeline."""

    codegen_pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute

    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
        super().__init__(root_op, tuner_ctx)

    @classmethod
    def supports_root_op(cls, root_op: ir.Operation) -> bool:
        return iree_codegen.isa_attention_op(root_op)

    def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
        return (
            rocm_constraint_generators.RocmAttentionVectorDistributeConstraintGenerator(
                self.get_op_info()
            )
        )

    def get_td_spec(
        self,
        config_list: list[common.TuningConfiguration],
    ) -> ir.Module:
        builder = spec_builder.AttentionSpecBuilder(self.get_op_info())
        return builder.build_td_spec(self._tuner_ctx, config_list)

    @classmethod
    def get_dispatch_kind(cls) -> common.DispatchKind:
        return common.DispatchKind.attention

    def get_knob_assignment(
        self,
        _config_list: list[common.TuningConfiguration],
    ) -> Optional[common.KnobAssignment]:
        return None
