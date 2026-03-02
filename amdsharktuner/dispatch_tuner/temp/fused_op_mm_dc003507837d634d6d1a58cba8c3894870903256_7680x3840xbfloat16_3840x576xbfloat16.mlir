module {
  func.func public @fused_op_mm_dc003507837d634d6d1a58cba8c3894870903256_7680x3840xbfloat16_3840x576xbfloat16(%arg0: !torch.vtensor<[7680,3840],bf16>, %arg1: !torch.vtensor<[3840,576],bf16>) -> !torch.vtensor<[7680,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[7680,3840],bf16>, !torch.vtensor<[3840,576],bf16> -> !torch.vtensor<[7680,576],bf16>
    return %0 : !torch.vtensor<[7680,576],bf16>
  }
}
