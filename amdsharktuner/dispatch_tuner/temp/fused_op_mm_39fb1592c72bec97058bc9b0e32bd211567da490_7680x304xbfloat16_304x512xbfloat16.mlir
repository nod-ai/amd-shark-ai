module {
  func.func public @fused_op_mm_39fb1592c72bec97058bc9b0e32bd211567da490_7680x304xbfloat16_304x512xbfloat16(%arg0: !torch.vtensor<[7680,304],bf16>, %arg1: !torch.vtensor<[304,512],bf16>) -> !torch.vtensor<[7680,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[7680,304],bf16>, !torch.vtensor<[304,512],bf16> -> !torch.vtensor<[7680,512],bf16>
    return %0 : !torch.vtensor<[7680,512],bf16>
  }
}
