module {
  func.func public @fused_op_mm_9dc2ac579101338abb0995f5ad38d09ac003851f_150000x4096xbfloat16_4096x4096xbfloat16(%arg0: !torch.vtensor<[150000,4096],bf16>, %arg1: !torch.vtensor<[4096,4096],bf16>) -> !torch.vtensor<[150000,4096],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,4096],bf16>, !torch.vtensor<[4096,4096],bf16> -> !torch.vtensor<[150000,4096],bf16>
    return %0 : !torch.vtensor<[150000,4096],bf16>
  }
}
