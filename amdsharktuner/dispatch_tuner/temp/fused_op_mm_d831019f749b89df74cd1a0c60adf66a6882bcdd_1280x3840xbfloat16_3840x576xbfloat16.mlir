module {
  func.func public @fused_op_mm_d831019f749b89df74cd1a0c60adf66a6882bcdd_1280x3840xbfloat16_3840x576xbfloat16(%arg0: !torch.vtensor<[1280,3840],bf16>, %arg1: !torch.vtensor<[3840,576],bf16>) -> !torch.vtensor<[1280,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1280,3840],bf16>, !torch.vtensor<[3840,576],bf16> -> !torch.vtensor<[1280,576],bf16>
    return %0 : !torch.vtensor<[1280,576],bf16>
  }
}
