module {
  func.func public @fused_op_mm_4da40c42a60f8073693bd03200dff3e8793230e6_16800000x128xbfloat16_128x134xbfloat16(%arg0: !torch.vtensor<[16800000,128],bf16>, %arg1: !torch.vtensor<[128,134],bf16>) -> !torch.vtensor<[16800000,134],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[16800000,128],bf16>, !torch.vtensor<[128,134],bf16> -> !torch.vtensor<[16800000,134],bf16>
    return %0 : !torch.vtensor<[16800000,134],bf16>
  }
}
