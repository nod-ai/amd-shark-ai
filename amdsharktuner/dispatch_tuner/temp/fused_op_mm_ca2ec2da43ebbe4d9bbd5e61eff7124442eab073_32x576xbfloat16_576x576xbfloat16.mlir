module {
  func.func public @fused_op_mm_ca2ec2da43ebbe4d9bbd5e61eff7124442eab073_32x576xbfloat16_576x576xbfloat16(%arg0: !torch.vtensor<[32,576],bf16>, %arg1: !torch.vtensor<[576,576],bf16>) -> !torch.vtensor<[32,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[32,576],bf16>, !torch.vtensor<[576,576],bf16> -> !torch.vtensor<[32,576],bf16>
    return %0 : !torch.vtensor<[32,576],bf16>
  }
}
