module {
  func.func public @fused_op_mm_98c67622c6def4077f3f3f1cc6fc90612dd7f641_11520x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[11520,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[11520,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[11520,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[11520,3840],bf16>
    return %0 : !torch.vtensor<[11520,3840],bf16>
  }
}
