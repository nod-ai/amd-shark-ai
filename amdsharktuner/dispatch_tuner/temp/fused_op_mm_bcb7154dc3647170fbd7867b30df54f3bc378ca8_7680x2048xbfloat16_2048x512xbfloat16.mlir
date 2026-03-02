module {
  func.func public @fused_op_mm_bcb7154dc3647170fbd7867b30df54f3bc378ca8_7680x2048xbfloat16_2048x512xbfloat16(%arg0: !torch.vtensor<[7680,2048],bf16>, %arg1: !torch.vtensor<[2048,512],bf16>) -> !torch.vtensor<[7680,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[7680,2048],bf16>, !torch.vtensor<[2048,512],bf16> -> !torch.vtensor<[7680,512],bf16>
    return %0 : !torch.vtensor<[7680,512],bf16>
  }
}
