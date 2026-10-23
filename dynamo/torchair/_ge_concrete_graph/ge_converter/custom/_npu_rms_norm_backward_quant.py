from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(2, 4, 64), F16(2, 4, 64), F32(2, 4), F16(64), F32(1)),
])
@register_fx_node_ge_converter(torch.ops.npu._npu_rms_norm_backward_quant.default)
def conveter_npu_rms_norm_backward_quant_default(
    dy: Tensor,
    x: Tensor,
    rstd: Tensor,
    gamma: Tensor,
    scale_x: Tensor,
    *,
    offset_x: Optional[Tensor] = None,
    div_mode: bool = True,
    quant_mode: str = "static",
    dst_type: int = 1,
    meta_outputs: List[TensorSpec] = None
):
    """NB: npu::_npu_rms_norm_backward_quant(Tensor dy, Tensor x, Tensor rstd, Tensor gamma, Tensor scale_x, *, Tensor? offset_x=None, bool? div_mode=True, str quant_mode="static", int? dst_type=1) -> (Tensor, Tensor)"""
    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    dx_out, dgamma_out = ge.RmsNormGradQuant(
        dy, x, rstd, gamma, scale_x, offset_x=offset_x,
        div_mode=div_mode, quant_mode=quant_mode, dst_type=acl_dst_type)
    return dx_out, dgamma_out
