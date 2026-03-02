module {
  func.func public @fused_op_bmm_da6bae89a6e2b82fee42ca200e7389032b139203_85x3x3xbfloat16_85x3x1152xbfloat16(%arg0: !torch.vtensor<[85,3,3],bf16>, %arg1: !torch.vtensor<[85,3,1152],bf16>) -> !torch.vtensor<[85,3,1152],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[85,3,3],bf16>, !torch.vtensor<[85,3,1152],bf16> -> !torch.vtensor<[85,3,1152],bf16>
    return %0 : !torch.vtensor<[85,3,1152],bf16>
  }
}
