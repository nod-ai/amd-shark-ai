module {
  func.func public @fused_op_mm_f95b35db09b9a9e8f4ea008692436b7f911edd41_4112x2048xbfloat16_2048x2048xbfloat16(%arg0: !torch.vtensor<[4112,2048],bf16>, %arg1: !torch.vtensor<[2048,2048],bf16>) -> !torch.vtensor<[4112,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4112,2048],bf16>, !torch.vtensor<[2048,2048],bf16> -> !torch.vtensor<[4112,2048],bf16>
    return %0 : !torch.vtensor<[4112,2048],bf16>
  }
}
