module {
  func.func public @fused_op_bmm_1a7564bf3b08185140366688ca1b76562becba68_128x3x3xbfloat16_128x3x2304xbfloat16(%arg0: !torch.vtensor<[128,3,3],bf16>, %arg1: !torch.vtensor<[128,3,2304],bf16>) -> !torch.vtensor<[128,3,2304],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[128,3,3],bf16>, !torch.vtensor<[128,3,2304],bf16> -> !torch.vtensor<[128,3,2304],bf16>
    return %0 : !torch.vtensor<[128,3,2304],bf16>
  }
}
