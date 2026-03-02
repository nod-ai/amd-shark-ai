module {
  func.func public @fused_op_mm_0b4492b63d99c96a5ec207ec465d266db4cc00d2_1280x576xbfloat16_576x576xbfloat16(%arg0: !torch.vtensor<[1280,576],bf16>, %arg1: !torch.vtensor<[576,576],bf16>) -> !torch.vtensor<[1280,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1280,576],bf16>, !torch.vtensor<[576,576],bf16> -> !torch.vtensor<[1280,576],bf16>
    return %0 : !torch.vtensor<[1280,576],bf16>
  }
}
