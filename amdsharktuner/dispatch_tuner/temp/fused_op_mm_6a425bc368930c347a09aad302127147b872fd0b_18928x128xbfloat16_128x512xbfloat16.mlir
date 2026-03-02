module {
  func.func public @fused_op_mm_6a425bc368930c347a09aad302127147b872fd0b_18928x128xbfloat16_128x512xbfloat16(%arg0: !torch.vtensor<[18928,128],bf16>, %arg1: !torch.vtensor<[128,512],bf16>) -> !torch.vtensor<[18928,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[18928,128],bf16>, !torch.vtensor<[128,512],bf16> -> !torch.vtensor<[18928,512],bf16>
    return %0 : !torch.vtensor<[18928,512],bf16>
  }
}
