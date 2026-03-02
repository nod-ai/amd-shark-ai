module {
  func.func public @fused_op_bmm_486d72565669702e4ff199fa22f699f0a931b1c0_5x384x192xbfloat16_5x192x384xbfloat16(%arg0: !torch.vtensor<[5,384,192],bf16>, %arg1: !torch.vtensor<[5,192,384],bf16>) -> !torch.vtensor<[5,384,384],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[5,384,192],bf16>, !torch.vtensor<[5,192,384],bf16> -> !torch.vtensor<[5,384,384],bf16>
    return %0 : !torch.vtensor<[5,384,384],bf16>
  }
}
