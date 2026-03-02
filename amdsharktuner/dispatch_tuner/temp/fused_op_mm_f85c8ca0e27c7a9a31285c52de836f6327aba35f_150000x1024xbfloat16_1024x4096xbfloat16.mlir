module {
  func.func public @fused_op_mm_f85c8ca0e27c7a9a31285c52de836f6327aba35f_150000x1024xbfloat16_1024x4096xbfloat16(%arg0: !torch.vtensor<[150000,1024],bf16>, %arg1: !torch.vtensor<[1024,4096],bf16>) -> !torch.vtensor<[150000,4096],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,1024],bf16>, !torch.vtensor<[1024,4096],bf16> -> !torch.vtensor<[150000,4096],bf16>
    return %0 : !torch.vtensor<[150000,4096],bf16>
  }
}
