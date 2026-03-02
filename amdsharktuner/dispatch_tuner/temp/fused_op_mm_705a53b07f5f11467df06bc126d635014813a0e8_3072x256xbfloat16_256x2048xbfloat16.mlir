module {
  func.func public @fused_op_mm_705a53b07f5f11467df06bc126d635014813a0e8_3072x256xbfloat16_256x2048xbfloat16(%arg0: !torch.vtensor<[3072,256],bf16>, %arg1: !torch.vtensor<[256,2048],bf16>) -> !torch.vtensor<[3072,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3072,256],bf16>, !torch.vtensor<[256,2048],bf16> -> !torch.vtensor<[3072,2048],bf16>
    return %0 : !torch.vtensor<[3072,2048],bf16>
  }
}
