module {
  func.func public @fused_op_mm_ebc3c2fe3d370c209036fbd34353270d8e8510c1_4352x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[4352,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[4352,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4352,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[4352,3840],bf16>
    return %0 : !torch.vtensor<[4352,3840],bf16>
  }
}
