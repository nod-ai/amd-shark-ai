module {
  func.func public @fused_op_mm_1a6f1657f751029f61922078ba2139da26ea01ad_150000x1024xbfloat16_1024x1024xbfloat16(%arg0: !torch.vtensor<[150000,1024],bf16>, %arg1: !torch.vtensor<[1024,1024],bf16>) -> !torch.vtensor<[150000,1024],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,1024],bf16>, !torch.vtensor<[1024,1024],bf16> -> !torch.vtensor<[150000,1024],bf16>
    return %0 : !torch.vtensor<[150000,1024],bf16>
  }
}
