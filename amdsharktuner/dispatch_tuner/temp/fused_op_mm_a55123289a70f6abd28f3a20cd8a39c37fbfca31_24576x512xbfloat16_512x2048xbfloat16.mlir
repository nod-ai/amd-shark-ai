module {
  func.func public @fused_op_mm_a55123289a70f6abd28f3a20cd8a39c37fbfca31_24576x512xbfloat16_512x2048xbfloat16(%arg0: !torch.vtensor<[24576,512],bf16>, %arg1: !torch.vtensor<[512,2048],bf16>) -> !torch.vtensor<[24576,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[24576,512],bf16>, !torch.vtensor<[512,2048],bf16> -> !torch.vtensor<[24576,2048],bf16>
    return %0 : !torch.vtensor<[24576,2048],bf16>
  }
}
