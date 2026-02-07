# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import os
from pathlib import Path
from typing import Callable


from fusilli_tuner.fusilli_tuner import (
    load_commands_from_file_or_args,
    build_compile_args,
)


@pytest.fixture
def tmp_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary files.

    Returns a callable that takes (content, suffix) and returns a Path to a
    temporary file with that content and suffix.
    """
    counter = 0

    def _create(content: str, suffix: str = ".txt") -> Path:
        nonlocal counter
        counter += 1
        temp_file = tmp_path / f"test_file_{counter}{suffix}"
        temp_file.write_text(content)
        return temp_file

    return _create


def test_load_fusilli_commands_from_file(tmp_file: Callable[[str, str], Path]) -> None:
    """Test loading Fusilli-specific commands from file."""
    content = """# Fusilli example commands
conv -F 1 --bf16 -n 1 -c 64 -H 28 -W 28 -k 128
matmul -M 1024 -N 1024 -K 1024 --a_type bf16
"""
    file_path = tmp_file(content, ".txt")

    result = load_commands_from_file_or_args(str(file_path), [])

    assert len(result) == 2
    assert result[0][0] == "conv"
    assert result[1][0] == "matmul"


def test_build_compile_args() -> None:
    """Test that build_compile_args filters unwanted flags and adds tuner flags."""
    compile_command = (
        "iree-compile --iree-hal-target-backends=rocm "
        "--iree-scheduling-dump-statistics-format=json "
        "--iree-scheduling-dump-statistics-file=/tmp/stats.json "
        "/path/to/input.mlir -o /path/to/output.vmfb"
    )
    benchmarks_dir = Path("/tmp/benchmarks")

    result = build_compile_args(compile_command, benchmarks_dir)

    # Statistics flags should be filtered out.
    assert "--iree-scheduling-dump-statistics-format" not in result
    assert "--iree-scheduling-dump-statistics-file" not in result
    assert "json" not in result
    assert "/tmp/stats.json" not in result
    # Original -o output should be filtered out.
    assert "/path/to/output.vmfb" not in result

    # Original flags should be preserved.
    assert "--iree-hal-target-backends=rocm" in result
    assert "/path/to/input.mlir" in result

    # Tuner-specific flags should be added.
    assert "--iree-config-add-tuner-attributes" in result
    assert "--iree-hal-dump-executable-benchmarks-to" in result
    assert str(benchmarks_dir) in result
    # Output should go to /dev/null.
    assert result[-2] == "-o"
    assert result[-1] == os.devnull
