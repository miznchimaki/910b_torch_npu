from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._utils.check_platform import is_arch35


@register_fx_node_ge_converter(torch.ops.npu.npu_mm_reduce_scatter_base.default)
def convert_npu_mm_reduce_scatter_base(
    self: Tensor,
    x2: Tensor,
    hcom: str,
    world_size: int,
    reduce_op: str = 'sum',
    bias: Optional[Tensor] = None,
    x1_scale: Optional[Tensor] = None,
    x2_scale: Optional[Tensor] = None,
    comm_turn: int = 0,
    output_dtype: Optional[int] = None,
    comm_mode: Optional[str] = None,
    meta_outputs: TensorSpec = None
):
    transpose_x1 = False
    transpose_x2 = False
    '''NB: npu::npu_mm_reduce_scatter_base(Tensor self, Tensor x2, str hcom, int world_size, *,
       str reduce_op='sum', Tensor? bias=None, Tensor? x1_scale=None, Tensor? x2_scale=None,
       int comm_turn=0, ScalarType? output_dtype=None, str? comm_mode=None) -> Tensor'''
    check_dtype(self, x2, bias=bias)
    check_valid_params(x1_scale=x1_scale, x2_scale=x2_scale, output_dtype=output_dtype)
    if not is_arch35():
        if comm_mode is not None:
            raise ValueError("The comm_mode parameter is not supported on Ascend910B/Ascend910C.")
        return ge.MatmulReduceScatter(self,
                                    x2,
                                    bias=bias,
                                    group=hcom,
                                    reduce_op=reduce_op,
                                    is_trans_a=transpose_x1,
                                    is_trans_b=transpose_x2,
                                    comm_turn=comm_turn,
                                    rank_size=world_size)
    else:
        if comm_mode is None:
            comm_mode = "ai_cpu"
        (out, amax_out) = ge.MatmulReduceScatterV2(self,
                                               x2,
                                               bias=bias,
                                               x1_scale=x1_scale,
                                               x2_scale=x2_scale,
                                               quant_scale=None,
                                               group=hcom,
                                               reduce_op=reduce_op,
                                               is_trans_a=transpose_x1,
                                               is_trans_b=transpose_x2,
                                               comm_turn=comm_turn,
                                               rank_size=world_size,
                                               y_dtype=self.dtype,
                                               comm_mode=comm_mode)
        return out


def check_dtype(x1: Tensor, x2: Tensor, bias: Optional[Tensor]):
    if x1.dtype != x2.dtype:
        raise AssertionError(f"Type of x1:{x1.dtype} and x2:{x2.dtype} must be same.")
    if (x1.dtype != DataType.DT_FLOAT16 and x1.dtype != DataType.DT_BF16):
        raise AssertionError(f"Input supported dtype is fp16/bf16, but got type {x1.dtype}.")

def check_valid_params(x1_scale: Optional[Tensor], x2_scale: Optional[Tensor], output_dtype: Optional[int]):
    if x1_scale is not None :
        raise ValueError("The graph mode does not support the x1_scale parameter currently. If required, \
                         please use eager mode or the graph mode of npu.npu_quant_mm_reduce_scatter API")
    if x2_scale is not None:
        raise ValueError("The graph mode does not support the x2_scale parameter currently. If required, \
                         please use eager mode or the graph mode of npu.npu_quant_mm_reduce_scatter API")
    if output_dtype is not None:
        raise ValueError("The graph mode does not support the output_dtype parameter currently. If required, \
                         please use eager mode or the graph mode of npu.npu_quant_mm_reduce_scatter API")
