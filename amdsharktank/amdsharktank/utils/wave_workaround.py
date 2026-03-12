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

This module automatically patches fx.Proxy.__getitem__ to make it context-aware,
so it only uses wave's override when an OpDispatcher context is active, otherwise
falls back to the original behavior.

See: https://github.com/iree-org/wave/commit/a62cf396
"""

from contextlib import contextmanager
from typing import Any
import functools


def _patch_fx_proxy_getitem():
    """Make fx.Proxy.__getitem__ context-aware to avoid requiring OpDispatcher.

    Wave's override requires an active OpDispatcher context. This patch makes the
    override conditional: it only uses wave's implementation when OpDispatcher.current()
    is available, otherwise it falls back to the original FX behavior.

    This allows both wave kernel tracing and normal PyTorch FX tracing to work.
    """
    try:
        import torch.fx as fx
    except ImportError:
        # If torch.fx is not available, there's nothing to patch
        return

    # Save the current __getitem__ (which might be wave's override or the original)
    current_getitem = getattr(fx.Proxy, "__getitem__", None)
    if current_getitem is None:
        return

    # Check if we've already patched it
    if hasattr(current_getitem, "_wave_workaround_patched"):
        return

    # Try to get the original fx.Proxy.__getitem__
    original_getitem = None

    # Check if we can find the original in the base class
    for base in fx.Proxy.__mro__[1:]:
        if hasattr(base, "__getitem__"):
            original_getitem = base.__getitem__
            break

    # If we couldn't find it in the MRO, create a fallback implementation
    if original_getitem is None:

        def original_getitem(self, key):
            """Fallback implementation that mimics original fx.Proxy.__getitem__."""
            from operator import getitem

            return self.tracer.create_proxy("call_function", getitem, (self, key), {})

    # Create a context-aware wrapper
    def context_aware_getitem(self, key):
        """Context-aware __getitem__ that uses wave's impl only when OpDispatcher is active."""
        try:
            # Try to get the OpDispatcher context
            from wave_lang.kernel.ops.base import OpDispatcher

            dispatcher = OpDispatcher.current()
            # If we got here, there's an active dispatcher, use wave's implementation
            return dispatcher.handle_getitem(self, key)
        except (ImportError, IndexError, AttributeError, LookupError):
            # No active OpDispatcher context (or wave not installed), use original behavior
            return original_getitem(self, key)

    # Mark as patched to avoid double-patching
    context_aware_getitem._wave_workaround_patched = True

    # Install our context-aware version
    fx.Proxy.__getitem__ = context_aware_getitem


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

    NOTE: This context manager is deprecated in favor of the automatic patching
    done by _patch_fx_proxy_getitem(), which makes the override context-aware
    globally instead of requiring explicit context managers.
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

    NOTE: This is now redundant with the global _patch_fx_proxy_getitem() patch,
    but kept for backwards compatibility.
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


# Apply the patches automatically when this module is imported
# The fx.Proxy patch makes the override context-aware globally
_patch_fx_proxy_getitem()
# The aot.export patch is kept for backwards compatibility but is now redundant
_patch_aot_export()


# Install an import hook to re-patch after wave_lang is imported
# This ensures our patch is applied even if wave overrides fx.Proxy.__getitem__ later
def _install_import_hook():
    """Install an import hook to re-patch fx.Proxy after wave_lang is imported."""
    import sys

    class WavePatchHook:
        """Meta path finder that patches fx.Proxy after wave_lang is imported."""

        def find_module(self, fullname, path=None):
            # Only interested in wave_lang
            if fullname == "wave_lang" or fullname.startswith("wave_lang."):
                return self
            return None

        def load_module(self, fullname):
            # Let the normal import machinery load the module
            if fullname in sys.modules:
                return sys.modules[fullname]

            # Find and execute the module normally
            import importlib.util

            spec = importlib.util.find_spec(fullname)
            if spec is None:
                raise ImportError(f"No module named {fullname}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            spec.loader.exec_module(module)

            # After wave_lang is loaded, re-patch fx.Proxy
            if fullname == "wave_lang":
                _patch_fx_proxy_getitem()

            return module

    # Install the hook
    sys.meta_path.insert(0, WavePatchHook())


try:
    _install_import_hook()
except Exception:
    # If the import hook fails, it's not critical
    # The initial patch should still work in most cases
    pass
