module {
  func.func public @fused_op_mm_660ea5711ba89f83ce0d70d8be3379f62ef688f7_5x4xbfloat16_4x384xbfloat16(%arg0: !torch.vtensor<[5,4],bf16>, %arg1: !torch.vtensor<[4,384],bf16>) -> !torch.vtensor<[5,384],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[5,4],bf16>, !torch.vtensor<[4,384],bf16> -> !torch.vtensor<[5,384],bf16>
    return %0 : !torch.vtensor<[5,384],bf16>
  }
}
