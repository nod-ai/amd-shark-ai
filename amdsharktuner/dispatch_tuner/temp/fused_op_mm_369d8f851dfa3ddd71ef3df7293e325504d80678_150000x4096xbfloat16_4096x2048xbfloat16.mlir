module {
  func.func public @fused_op_mm_369d8f851dfa3ddd71ef3df7293e325504d80678_150000x4096xbfloat16_4096x2048xbfloat16(%arg0: !torch.vtensor<[150000,4096],bf16>, %arg1: !torch.vtensor<[4096,2048],bf16>) -> !torch.vtensor<[150000,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,4096],bf16>, !torch.vtensor<[4096,2048],bf16> -> !torch.vtensor<[150000,2048],bf16>
    return %0 : !torch.vtensor<[150000,2048],bf16>
  }
}
