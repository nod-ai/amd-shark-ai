module {
  func.func public @fused_op_mm_f7ac7853dda987b6312b2688b49974f8728f2f7c_21760x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[21760,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[21760,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[21760,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[21760,3840],bf16>
    return %0 : !torch.vtensor<[21760,3840],bf16>
  }
}
