from typing import List

import torch
from torch.fx.node import has_side_effect

from ._lib import lib
from ._scope_attr import scope_enter, scope_exit


if not hasattr(torch.ops.npugraph_ex, "scope_enter"):
    lib.define(
        """
        scope_enter(str[] keys, str[] values, bool need_execute=False) -> None
        """
    )
    has_side_effect(torch.ops.npugraph_ex.scope_enter.default)


    @torch.library.impl(lib, "scope_enter", "Meta")
    def kernel_meta(keys: List[str], values: List[str], need_execute=False):
        if need_execute:
            scope_enter(keys, values)


    def kernel_impl(keys: List[str], values: List[str], need_execute=False):
        if need_execute:
            scope_enter(keys, values)


    torch.library.impl(lib, "scope_enter", "CPU")(kernel_impl)
    torch.library.impl(lib, "scope_enter", "PrivateUse1")(kernel_impl)
    torch.library.impl(lib, "scope_enter", "BackendSelect")(kernel_impl)


@torch.compiler.allow_in_graph
def _core_limit_scope_enter(stream, keys, values):
    # The npugraph_ex pre-AOT pass resolves the Stream proxy and rewrites this
    # marker to scope_enter with stream identity stored in the same node.
    return torch.ops.npugraph_ex.scope_enter(keys, values)


# The source can be loaded as different Python function objects through
# torch_npu.dynamo.npugraph_ex and npugraph_ex. Match it by semantic metadata.
_core_limit_scope_enter._is_core_limit_scope_marker = True


# Keep the marker until replace_stream_event_pass resolves the Stream proxy.
has_side_effect(_core_limit_scope_enter)


def _npu_scope_enter(attrs):
    keys, values = zip(*attrs)
    keys, values = list(keys), list(values)
    return torch.ops.npugraph_ex.scope_enter(keys, values)


def _npu_core_limit_scope_enter(stream, attrs):
    keys, values = zip(*attrs)
    return _core_limit_scope_enter(stream, list(keys), list(values))


if not hasattr(torch.ops.npugraph_ex, "scope_exit"):
    lib.define(
        """
        scope_exit(bool need_execute=False) -> None
        """
    )
    has_side_effect(torch.ops.npugraph_ex.scope_exit.default)


    @torch.library.impl(lib, "scope_exit", "Meta")
    def kernel_meta(need_execute=False):
        if need_execute:
            scope_exit()


    def kernel_impl(need_execute=False):
        if need_execute:
            scope_exit()


    torch.library.impl(lib, "scope_exit", "CPU")(kernel_impl)
    torch.library.impl(lib, "scope_exit", "PrivateUse1")(kernel_impl)
    torch.library.impl(lib, "scope_exit", "BackendSelect")(kernel_impl)


def _npu_scope_exit():
    return torch.ops.npugraph_ex.scope_exit()
