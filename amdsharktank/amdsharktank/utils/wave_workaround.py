# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Workaround for wave's fx.Proxy.__getitem__ override that breaks torch.export.

Wave's commit a62cf396 added a global override to fx.Proxy.__getitem__ that
unconditionally requires an active OpDispatcher context. This breaks PyTorch's
export which uses FX tracing outside of wave's kernel context.

This module automatically patches iree.turbine.aot.export to restore the original
fx.Proxy.__getitem__ behavior during export operations.

See: https://github.com/iree-org/wave/commit/a62cf396
"""

from contextlib import contextmanager
from typing import Any
import functools


@contextmanager
def restore_fx_proxy_getitem():
    """Context manager that temporarily restores the original fx.Proxy.__getitem__.

    Wave overrides fx.Proxy.__getitem__ with a custom implementation that requires
    an active OpDispatcher context. This breaks PyTorch's export/tracing when used
    outside of wave kernel contexts.

    This context manager saves wave's override, restores the original behavior,
    and then restores wave's override when done.

    Usage:
        with restore_fx_proxy_getitem():
            # torch.export operations that would fail with wave's override
            exported = torch.export.export(model, ...)
    """
    try:
        import torch.fx as fx
    except ImportError:
        # If torch.fx is not available, there's nothing to patch
        yield
        return

    # Save wave's current override (if it exists)
    wave_getitem = getattr(fx.Proxy, "__getitem__", None)

    # Try to get the original fx.Proxy.__getitem__ from the method itself
    # If wave has already overridden it, we need to find the original implementation
    original_getitem = None

    # Check if we can find the original in the base class or somewhere else
    # The original implementation is typically a simple method that creates a call_function node
    if wave_getitem is not None:
        # Try to find the original implementation by looking at the MRO
        for base in fx.Proxy.__mro__[1:]:
            if hasattr(base, "__getitem__"):
                original_getitem = base.__getitem__
                break

        # If we couldn't find it in the MRO, create a fallback implementation
        # that matches the original fx.Proxy behavior
        if original_getitem is None:

            def original_getitem(self, key):
                """Fallback implementation that mimics original fx.Proxy.__getitem__."""
                # The original fx.Proxy.__getitem__ creates a call_function node
                # for operator.getitem. We'll delegate to the tracer.
                from operator import getitem

                return self.tracer.create_proxy(
                    "call_function", getitem, (self, key), {}
                )

    # Temporarily restore the original
    if original_getitem is not None:
        fx.Proxy.__getitem__ = original_getitem

    try:
        yield
    finally:
        # Restore wave's override
        if wave_getitem is not None:
            fx.Proxy.__getitem__ = wave_getitem


def _patch_aot_export():
    """Automatically patch iree.turbine.aot.export to work around wave's fx.Proxy override.

    This function wraps aot.export so that it temporarily restores the original
    fx.Proxy.__getitem__ behavior during export, then restores wave's override afterward.

    This is done automatically when this module is imported, so no code changes are needed
    in the rest of the codebase.
    """
    try:
        from iree.turbine import aot
    except ImportError:
        # If iree.turbine is not available, skip patching
        return

    # Save the original export function
    _original_aot_export = aot.export

    @functools.wraps(_original_aot_export)
    def _patched_aot_export(*args, **kwargs):
        """Patched version of aot.export that works around wave's fx.Proxy override."""
        with restore_fx_proxy_getitem():
            return _original_aot_export(*args, **kwargs)

    # Replace aot.export with our patched version
    aot.export = _patched_aot_export


# Apply the patch automatically when this module is imported
_patch_aot_export()
