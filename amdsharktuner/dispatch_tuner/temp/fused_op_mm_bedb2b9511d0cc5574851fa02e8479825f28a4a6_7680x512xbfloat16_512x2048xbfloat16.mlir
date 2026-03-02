module {
  func.func public @fused_op_mm_bedb2b9511d0cc5574851fa02e8479825f28a4a6_7680x512xbfloat16_512x2048xbfloat16(%arg0: !torch.vtensor<[7680,512],bf16>, %arg1: !torch.vtensor<[512,2048],bf16>) -> !torch.vtensor<[7680,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[7680,512],bf16>, !torch.vtensor<[512,2048],bf16> -> !torch.vtensor<[7680,2048],bf16>
    return %0 : !torch.vtensor<[7680,2048],bf16>
  }
}
