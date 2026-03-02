module {
  func.func public @fused_op_mm_5bd10678f76842e3ee4f78d7c7f0c8883878f889_3670016x128xbfloat16_128x128xbfloat16(%arg0: !torch.vtensor<[3670016,128],bf16>, %arg1: !torch.vtensor<[128,128],bf16>) -> !torch.vtensor<[3670016,128],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3670016,128],bf16>, !torch.vtensor<[128,128],bf16> -> !torch.vtensor<[3670016,128],bf16>
    return %0 : !torch.vtensor<[3670016,128],bf16>
  }
}
