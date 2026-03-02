module {
  func.func public @fused_op_bmm_65b38131d28a00193187ca5244d4b1feba490812_5x384x384xbfloat16_5x384x192xbfloat16(%arg0: !torch.vtensor<[5,384,384],bf16>, %arg1: !torch.vtensor<[5,384,192],bf16>) -> !torch.vtensor<[5,384,192],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[5,384,384],bf16>, !torch.vtensor<[5,384,192],bf16> -> !torch.vtensor<[5,384,192],bf16>
    return %0 : !torch.vtensor<[5,384,192],bf16>
  }
}
