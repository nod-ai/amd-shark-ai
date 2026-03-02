module {
  func.func public @fused_op_mm_802788ed6f6fa79a7b4bd8492e2f1e320bc5ba41_4096x8192xbfloat16_8192x2048xbfloat16(%arg0: !torch.vtensor<[4096,8192],bf16>, %arg1: !torch.vtensor<[8192,2048],bf16>) -> !torch.vtensor<[4096,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4096,8192],bf16>, !torch.vtensor<[8192,2048],bf16> -> !torch.vtensor<[4096,2048],bf16>
    return %0 : !torch.vtensor<[4096,2048],bf16>
  }
}
