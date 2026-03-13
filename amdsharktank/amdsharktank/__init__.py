# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util

msg = """No module named 'torch'. Follow https://pytorch.org/get-started/locally/#start-locally to install 'torch'.
For example, on Linux to install with CPU support run:
  pip3 install torch --index-url https://download.pytorch.org/whl/cpu
"""

if spec := importlib.util.find_spec("torch") is None:
    raise ModuleNotFoundError(msg)

# Apply wave workaround patches early
# This patches iree.turbine.aot.export to work around wave's fx.Proxy.__getitem__ override
# See: amdsharktank/utils/wave_workaround.py for details
try:
    from .utils import wave_workaround  # noqa: F401
except ImportError:
    # If imports fail, the workaround will be applied when the module is eventually imported
    pass
