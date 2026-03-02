module {
  func.func public @fused_op_mm_334ac0f6d37585eb819016011650a9537b681dd3_21760x20xbfloat16_20x3840xbfloat16(%arg0: !torch.vtensor<[21760,20],bf16>, %arg1: !torch.vtensor<[20,3840],bf16>) -> !torch.vtensor<[21760,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[21760,20],bf16>, !torch.vtensor<[20,3840],bf16> -> !torch.vtensor<[21760,3840],bf16>
    return %0 : !torch.vtensor<[21760,3840],bf16>
  }
}
