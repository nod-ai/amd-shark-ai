module {
  func.func public @fused_op_mm_3bae4c2bfbe0ebbc639738b4ac09969f231dd821_4096x2048xbfloat16_2048x2048xbfloat16(%arg0: !torch.vtensor<[4096,2048],bf16>, %arg1: !torch.vtensor<[2048,2048],bf16>) -> !torch.vtensor<[4096,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4096,2048],bf16>, !torch.vtensor<[2048,2048],bf16> -> !torch.vtensor<[4096,2048],bf16>
    return %0 : !torch.vtensor<[4096,2048],bf16>
  }
}
