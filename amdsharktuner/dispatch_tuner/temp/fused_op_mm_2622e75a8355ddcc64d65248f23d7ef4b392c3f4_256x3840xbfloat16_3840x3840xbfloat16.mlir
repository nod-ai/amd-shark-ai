module {
  func.func public @fused_op_mm_2622e75a8355ddcc64d65248f23d7ef4b392c3f4_256x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[256,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[256,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[256,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[256,3840],bf16>
    return %0 : !torch.vtensor<[256,3840],bf16>
  }
}
