module {
  func.func public @fused_op_mm_60df32bc4bc84f3b47640b6206a19e4b9ac8081f_5x384xbfloat16_384x384xbfloat16(%arg0: !torch.vtensor<[5,384],bf16>, %arg1: !torch.vtensor<[384,384],bf16>) -> !torch.vtensor<[5,384],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[5,384],bf16>, !torch.vtensor<[384,384],bf16> -> !torch.vtensor<[5,384],bf16>
    return %0 : !torch.vtensor<[5,384],bf16>
  }
}
