module {
  func.func public @fused_op_mm_7f341226f1e607e75e905569e7fc4f215a9a0136_32768x128xbfloat16_128x128xbfloat16(%arg0: !torch.vtensor<[32768,128],bf16>, %arg1: !torch.vtensor<[128,128],bf16>) -> !torch.vtensor<[32768,128],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[32768,128],bf16>, !torch.vtensor<[128,128],bf16> -> !torch.vtensor<[32768,128],bf16>
    return %0 : !torch.vtensor<[32768,128],bf16>
  }
}
