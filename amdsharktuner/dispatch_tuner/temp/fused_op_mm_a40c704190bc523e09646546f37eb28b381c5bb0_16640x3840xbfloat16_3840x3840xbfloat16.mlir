module {
  func.func public @fused_op_mm_a40c704190bc523e09646546f37eb28b381c5bb0_16640x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[16640,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[16640,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[16640,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[16640,3840],bf16>
    return %0 : !torch.vtensor<[16640,3840],bf16>
  }
}
