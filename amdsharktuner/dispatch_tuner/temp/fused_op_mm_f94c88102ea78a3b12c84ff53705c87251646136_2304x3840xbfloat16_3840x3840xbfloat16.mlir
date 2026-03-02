module {
  func.func public @fused_op_mm_f94c88102ea78a3b12c84ff53705c87251646136_2304x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[2304,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[2304,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2304,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[2304,3840],bf16>
    return %0 : !torch.vtensor<[2304,3840],bf16>
  }
}
