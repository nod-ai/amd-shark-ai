module {
  func.func public @fused_op_mm_815596f93e67e8e8bdaa1bd20990c050aacfec1e_4096x1152xbfloat16_1152x576xbfloat16(%arg0: !torch.vtensor<[4096,1152],bf16>, %arg1: !torch.vtensor<[1152,576],bf16>) -> !torch.vtensor<[4096,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4096,1152],bf16>, !torch.vtensor<[1152,576],bf16> -> !torch.vtensor<[4096,576],bf16>
    return %0 : !torch.vtensor<[4096,576],bf16>
  }
}
