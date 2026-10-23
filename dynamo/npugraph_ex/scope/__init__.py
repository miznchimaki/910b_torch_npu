import torch

from ._scope import _npu_core_limit_scope_enter, _npu_scope_enter, _npu_scope_exit


class _Scope:
    def __init__(self, attrs):
        self.attrs = attrs

    def __enter__(self):
        return _npu_scope_enter(self.attrs)

    def __exit__(self, *args):
        return _npu_scope_exit()


class _CoreLimitScope:
    def __init__(self, attrs, stream=None):
        self.attrs = attrs
        self.stream = stream

    def __enter__(self):
        if self.stream is None:
            return _npu_scope_enter(self.attrs)
        # Keep the explicit stream and core attributes in one Dynamo node.
        return _npu_core_limit_scope_enter(self.stream, self.attrs)

    def __exit__(self, *args):
        return _npu_scope_exit()


class _EagerLimitCore:
    def __init__(self, cube_num: int, vector_num: int, stream=None):
        # Defer default-stream lookup until scope entry to match with semantics.
        self.cube_num = cube_num
        self.vector_num = vector_num
        self.stream = stream
        self.current_stream = None
        self.origin_cube_num = -1
        self.origin_vector_num = -1

    def __enter__(self):
        # Freeze the target stream for both setting and later restoration.
        self.current_stream = self.stream if self.stream is not None else torch.npu.current_stream()
        # Save the target stream's original limits before applying new values.
        origin = torch.npu.get_stream_limit(self.current_stream)
        self.origin_cube_num = origin["cube_core_num"]
        self.origin_vector_num = origin["vector_core_num"]
        torch.npu.set_stream_limit(self.current_stream, self.cube_num, self.vector_num)
        return self

    def __exit__(self, *args):
        # Restore the same stream even if the body switched current streams.
        torch.npu.set_stream_limit(self.current_stream, self.origin_cube_num, self.origin_vector_num)


def limit_core_num(op_aicore_num: int, op_vectorcore_num: int, stream=None):
    # Eager mode performs get/set immediately through the context manager.
    if not torch.compiler.is_compiling():
        return _EagerLimitCore(op_aicore_num, op_vectorcore_num, stream)

    # The pre-AOT marker carries the explicit stream and core attributes
    # together, avoiding a positional association between separate FX nodes.
    return _CoreLimitScope([
        ("_op_aicore_num", str(op_aicore_num)),
        ("_op_vectorcore_num", str(op_vectorcore_num))
    ], stream)
