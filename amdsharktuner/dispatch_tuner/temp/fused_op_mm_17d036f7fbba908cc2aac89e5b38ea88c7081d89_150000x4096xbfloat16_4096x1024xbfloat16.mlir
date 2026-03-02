module {
  func.func public @fused_op_mm_17d036f7fbba908cc2aac89e5b38ea88c7081d89_150000x4096xbfloat16_4096x1024xbfloat16(%arg0: !torch.vtensor<[150000,4096],bf16>, %arg1: !torch.vtensor<[4096,1024],bf16>) -> !torch.vtensor<[150000,1024],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,4096],bf16>, !torch.vtensor<[4096,1024],bf16> -> !torch.vtensor<[150000,1024],bf16>
    return %0 : !torch.vtensor<[150000,1024],bf16>
  }
}
