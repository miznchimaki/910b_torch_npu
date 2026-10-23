from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._utils.check_platform import is_arch35

@register_fx_node_ge_converter(torch.ops.npu.npu_all_gather_base_mm.default)
def convert_npu_all_gather_base_mm(
    self: Tensor,
    x2: Tensor,
    hcom: str,
    world_size: int,
    bias: Optional[Tensor] = None,
    x1_scale: Optional[Tensor] = None,
    x2_scale: Optional[Tensor] = None,
    gather_index: int = 0,
    gather_output: bool = True,
    comm_turn: int = 0,
    output_dtype: Optional[int] = None,
    comm_mode: Optional[str] = None,
    meta_outputs: TensorSpec = None
):
    transpose_x1 = False
    transpose_x2 = False
    '''
    NB: npu_all_gather_base_mm(Tensor self, Tensor x2, str hcom, int world_size, *, Tensor? bias=None, Tensor? x1_scale=None,
    Tensor? x2_scale=None, int gather_index=0, bool gather_output=True, int comm_turn=0, ScalarType? output_dtype=None, str? comm_mode=None) -> (Tensor, Tensor)
    '''
    check_dtype(self, x2, bias=bias)
    if x1_scale is not None:
        raise ValueError(f"The graph mode does not support the x1_scale parameter. If required, please use the eager mode or the graph mode of the npu_all_gather_quant_mm API.")
    if x2_scale is not None:
        raise ValueError(f"The graph mode does not support the x2_scale parameter. If required, please use the eager mode or the graph mode of the npu_all_gather_quant_mm API.")
    if output_dtype is not None:
        raise ValueError(f"The graph mode does not support the output_dtype parameter. If required, please use the eager mode or the graph mode of the npu_all_gather_quant_mm API.")
    if not is_arch35():
        if comm_mode is not None:
            raise ValueError(f"The graph mode does not support the comm_mode parameter. If required, please use the eager mode or the graph mode of the npu_all_gather_quant_mm API.")
        return ge.AllGatherMatmul(self,
                              x2,
                              bias=bias,
                              group=hcom,
                              gather_index=gather_index,
                              is_trans_a=transpose_x1,
                              is_trans_b=transpose_x2,
                              comm_turn=comm_turn,
                              rank_size=world_size,
                              is_gather_out=gather_output)
    else:
        (out, gather_out, amax_out) = ge.AllGatherMatmulV2(self,
                                                       x2,
                                                       bias=bias,
                                                       x1_scale=x1_scale,
                                                       x2_scale=x2_scale,
                                                       quant_scale=None,
                                                       group=hcom,
                                                       is_trans_a=transpose_x1,
                                                       is_trans_b=transpose_x2,
                                                       gather_index=gather_index,
                                                       comm_turn=comm_turn,
                                                       rank_size=world_size,
                                                       block_size=0,
                                                       group_size=0,
                                                       is_gather_out=gather_output,
                                                       is_amax_out=False,
                                                       y_dtype=self.dtype,
                                                       comm_mode=comm_mode)
        return (out, gather_out)


def check_dtype(x1: Tensor, x2: Tensor, bias: Optional[Tensor]):
    if x1.dtype != x2.dtype:
        raise AssertionError(f"Type of x1:{x1.dtype} and x2:{x2.dtype} must be same.")
    if (x1.dtype != DataType.DT_FLOAT16 and x1.dtype != DataType.DT_BF16):
        raise AssertionError(f"Input supported dtype is fp16/bf16, but got type {x1.dtype}.")
