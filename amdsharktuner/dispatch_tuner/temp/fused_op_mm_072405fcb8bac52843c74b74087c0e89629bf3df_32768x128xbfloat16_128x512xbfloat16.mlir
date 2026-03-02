module {
  func.func public @fused_op_mm_072405fcb8bac52843c74b74087c0e89629bf3df_32768x128xbfloat16_128x512xbfloat16(%arg0: !torch.vtensor<[32768,128],bf16>, %arg1: !torch.vtensor<[128,512],bf16>) -> !torch.vtensor<[32768,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[32768,128],bf16>, !torch.vtensor<[128,512],bf16> -> !torch.vtensor<[32768,512],bf16>
    return %0 : !torch.vtensor<[32768,512],bf16>
  }
}
