module {
  func.func public @fused_op_mm_b2be1b7a2b864a6dba0bc3d5d146ff69c036d353_1280x1728xbfloat16_1728x576xbfloat16(%arg0: !torch.vtensor<[1280,1728],bf16>, %arg1: !torch.vtensor<[1728,576],bf16>) -> !torch.vtensor<[1280,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1280,1728],bf16>, !torch.vtensor<[1728,576],bf16> -> !torch.vtensor<[1280,576],bf16>
    return %0 : !torch.vtensor<[1280,576],bf16>
  }
}
