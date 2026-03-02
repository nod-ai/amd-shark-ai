module {
  func.func public @fused_op_mm_e2e72c2e0c1af04b06aeee2cfe0e751bcb850a6c_24576x2048xbfloat16_2048x512xbfloat16(%arg0: !torch.vtensor<[24576,2048],bf16>, %arg1: !torch.vtensor<[2048,512],bf16>) -> !torch.vtensor<[24576,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[24576,2048],bf16>, !torch.vtensor<[2048,512],bf16> -> !torch.vtensor<[24576,512],bf16>
    return %0 : !torch.vtensor<[24576,512],bf16>
  }
}
