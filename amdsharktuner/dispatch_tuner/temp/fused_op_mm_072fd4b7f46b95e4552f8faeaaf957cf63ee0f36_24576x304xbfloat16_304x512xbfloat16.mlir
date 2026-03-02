module {
  func.func public @fused_op_mm_072fd4b7f46b95e4552f8faeaaf957cf63ee0f36_24576x304xbfloat16_304x512xbfloat16(%arg0: !torch.vtensor<[24576,304],bf16>, %arg1: !torch.vtensor<[304,512],bf16>) -> !torch.vtensor<[24576,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[24576,304],bf16>, !torch.vtensor<[304,512],bf16> -> !torch.vtensor<[24576,512],bf16>
    return %0 : !torch.vtensor<[24576,512],bf16>
  }
}
