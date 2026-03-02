module {
  func.func public @fused_op_mm_ae2a977847148757becb96c80bd61bd684d10f48_6400x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[6400,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[6400,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[6400,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[6400,3840],bf16>
    return %0 : !torch.vtensor<[6400,3840],bf16>
  }
}
