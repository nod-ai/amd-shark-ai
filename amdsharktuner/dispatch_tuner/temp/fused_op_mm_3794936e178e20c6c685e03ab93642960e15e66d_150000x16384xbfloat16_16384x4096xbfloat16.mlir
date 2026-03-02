module {
  func.func public @fused_op_mm_3794936e178e20c6c685e03ab93642960e15e66d_150000x16384xbfloat16_16384x4096xbfloat16(%arg0: !torch.vtensor<[150000,16384],bf16>, %arg1: !torch.vtensor<[16384,4096],bf16>) -> !torch.vtensor<[150000,4096],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,16384],bf16>, !torch.vtensor<[16384,4096],bf16> -> !torch.vtensor<[150000,4096],bf16>
    return %0 : !torch.vtensor<[150000,4096],bf16>
  }
}
