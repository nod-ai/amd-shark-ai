module {
  func.func public @fused_op_mm_ab4d8152dba7dbdd3cc67c029d36f86b87c85f36_4096x6144xbfloat16_6144x2048xbfloat16(%arg0: !torch.vtensor<[4096,6144],bf16>, %arg1: !torch.vtensor<[6144,2048],bf16>) -> !torch.vtensor<[4096,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4096,6144],bf16>, !torch.vtensor<[6144,2048],bf16> -> !torch.vtensor<[4096,2048],bf16>
    return %0 : !torch.vtensor<[4096,2048],bf16>
  }
}
