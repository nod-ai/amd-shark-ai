module {
  func.func public @fused_op_mm_1ef0a140fbeffa95e785fbee93412ee3e2083a66_7680x576xbfloat16_576x3840xbfloat16(%arg0: !torch.vtensor<[7680,576],bf16>, %arg1: !torch.vtensor<[576,3840],bf16>) -> !torch.vtensor<[7680,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[7680,576],bf16>, !torch.vtensor<[576,3840],bf16> -> !torch.vtensor<[7680,3840],bf16>
    return %0 : !torch.vtensor<[7680,3840],bf16>
  }
}
