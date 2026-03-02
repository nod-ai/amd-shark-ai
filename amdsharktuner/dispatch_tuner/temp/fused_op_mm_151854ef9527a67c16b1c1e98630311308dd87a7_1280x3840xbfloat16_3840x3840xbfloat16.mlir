module {
  func.func public @fused_op_mm_151854ef9527a67c16b1c1e98630311308dd87a7_1280x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[1280,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[1280,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1280,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[1280,3840],bf16>
    return %0 : !torch.vtensor<[1280,3840],bf16>
  }
}
