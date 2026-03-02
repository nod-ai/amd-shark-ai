module {
  func.func public @fused_op_mm_2a5ff69bcc92a4834948ecb430fce119f3a3ce74_18928x128xbfloat16_128x128xbfloat16(%arg0: !torch.vtensor<[18928,128],bf16>, %arg1: !torch.vtensor<[128,128],bf16>) -> !torch.vtensor<[18928,128],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[18928,128],bf16>, !torch.vtensor<[128,128],bf16> -> !torch.vtensor<[18928,128],bf16>
    return %0 : !torch.vtensor<[18928,128],bf16>
  }
}
