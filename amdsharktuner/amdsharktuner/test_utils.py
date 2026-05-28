# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import pytest

from collections.abc import Sequence
from typing import Generator
from logging import Logger
from unittest.mock import MagicMock
from typing import Any

from iree.compiler import ir  # type: ignore
from iree.compiler.dialects import iree_gpu  # type: ignore

from amdsharktuner import common


@pytest.fixture
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    mock_logger = MagicMock(spec=Logger)
    with common.TunerContext(logger=mock_logger) as ctx:
        yield ctx


@pytest.fixture
def mlir_ctx() -> Generator[ir.Context, None, None]:
    with ir.Context() as ctx:
        yield ctx


def get_test_lowering_config(
    tuner_ctx: common.TunerContext,
    **kwargs: Any,
) -> iree_gpu.LoweringConfigAttr:
    lowering_config_dict: dict[str, Any] = {}
    for key, value in kwargs.items():
        promoted_value = value
        match key:
            case "workgroup" | "reduction" | "subgroup" | "promote_operands" | "padding" | "padding_conv":
                if isinstance(value, Sequence):
                    promoted_value = ir.ArrayAttr.get(
                        [tuner_ctx.type.getI64(x) for x in value]
                    )
            case "subgroup_basis":
                counts, mapping = value
                counts_attr = tuner_ctx.type.getI64ArrayAttr(counts)
                mapping_attr = tuner_ctx.type.getI64ArrayAttr(mapping)
                promoted_value = ir.ArrayAttr.get([counts_attr, mapping_attr])
            case "mma_kind":
                pass
            case _:
                raise AssertionError(f"Unhandled key in lowering configuration: {key}")

        lowering_config_dict[key] = promoted_value
    return iree_gpu.LoweringConfigAttr.get(ir.DictAttr.get(lowering_config_dict))
