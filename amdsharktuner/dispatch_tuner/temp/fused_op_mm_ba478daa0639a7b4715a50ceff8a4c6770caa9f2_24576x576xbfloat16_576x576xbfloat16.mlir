module {
  func.func public @fused_op_mm_ba478daa0639a7b4715a50ceff8a4c6770caa9f2_24576x576xbfloat16_576x576xbfloat16(%arg0: !torch.vtensor<[24576,576],bf16>, %arg1: !torch.vtensor<[576,576],bf16>) -> !torch.vtensor<[24576,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[24576,576],bf16>, !torch.vtensor<[576,576],bf16> -> !torch.vtensor<[24576,576],bf16>
    return %0 : !torch.vtensor<[24576,576],bf16>
  }
}
