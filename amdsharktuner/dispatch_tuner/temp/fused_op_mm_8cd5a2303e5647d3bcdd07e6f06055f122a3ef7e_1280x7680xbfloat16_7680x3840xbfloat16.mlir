module {
  func.func public @fused_op_mm_8cd5a2303e5647d3bcdd07e6f06055f122a3ef7e_1280x7680xbfloat16_7680x3840xbfloat16(%arg0: !torch.vtensor<[1280,7680],bf16>, %arg1: !torch.vtensor<[7680,3840],bf16>) -> !torch.vtensor<[1280,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1280,7680],bf16>, !torch.vtensor<[7680,3840],bf16> -> !torch.vtensor<[1280,3840],bf16>
    return %0 : !torch.vtensor<[1280,3840],bf16>
  }
}
