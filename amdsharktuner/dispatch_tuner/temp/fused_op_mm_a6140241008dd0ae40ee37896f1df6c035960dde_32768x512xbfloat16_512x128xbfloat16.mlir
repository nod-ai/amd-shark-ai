module {
  func.func public @fused_op_mm_a6140241008dd0ae40ee37896f1df6c035960dde_32768x512xbfloat16_512x128xbfloat16(%arg0: !torch.vtensor<[32768,512],bf16>, %arg1: !torch.vtensor<[512,128],bf16>) -> !torch.vtensor<[32768,128],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[32768,512],bf16>, !torch.vtensor<[512,128],bf16> -> !torch.vtensor<[32768,128],bf16>
    return %0 : !torch.vtensor<[32768,128],bf16>
  }
}
