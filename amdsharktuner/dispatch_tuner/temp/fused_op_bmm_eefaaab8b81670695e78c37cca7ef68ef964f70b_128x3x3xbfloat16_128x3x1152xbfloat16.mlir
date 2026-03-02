module {
  func.func public @fused_op_bmm_eefaaab8b81670695e78c37cca7ef68ef964f70b_128x3x3xbfloat16_128x3x1152xbfloat16(%arg0: !torch.vtensor<[128,3,3],bf16>, %arg1: !torch.vtensor<[128,3,1152],bf16>) -> !torch.vtensor<[128,3,1152],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[128,3,3],bf16>, !torch.vtensor<[128,3,1152],bf16> -> !torch.vtensor<[128,3,1152],bf16>
    return %0 : !torch.vtensor<[128,3,1152],bf16>
  }
}
