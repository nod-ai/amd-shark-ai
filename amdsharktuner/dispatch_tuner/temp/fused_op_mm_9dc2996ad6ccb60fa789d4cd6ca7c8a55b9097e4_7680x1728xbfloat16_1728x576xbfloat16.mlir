module {
  func.func public @fused_op_mm_9dc2996ad6ccb60fa789d4cd6ca7c8a55b9097e4_7680x1728xbfloat16_1728x576xbfloat16(%arg0: !torch.vtensor<[7680,1728],bf16>, %arg1: !torch.vtensor<[1728,576],bf16>) -> !torch.vtensor<[7680,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[7680,1728],bf16>, !torch.vtensor<[1728,576],bf16> -> !torch.vtensor<[7680,576],bf16>
    return %0 : !torch.vtensor<[7680,576],bf16>
  }
}
