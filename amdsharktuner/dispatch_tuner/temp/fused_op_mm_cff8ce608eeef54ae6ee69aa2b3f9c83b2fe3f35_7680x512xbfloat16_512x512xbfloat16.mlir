module {
  func.func public @fused_op_mm_cff8ce608eeef54ae6ee69aa2b3f9c83b2fe3f35_7680x512xbfloat16_512x512xbfloat16(%arg0: !torch.vtensor<[7680,512],bf16>, %arg1: !torch.vtensor<[512,512],bf16>) -> !torch.vtensor<[7680,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[7680,512],bf16>, !torch.vtensor<[512,512],bf16> -> !torch.vtensor<[7680,512],bf16>
    return %0 : !torch.vtensor<[7680,512],bf16>
  }
}
