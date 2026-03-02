module {
  func.func public @fused_op_mm_8376f86f567a41570cba94404136af5c9eb0ea90_165x576xbfloat16_576x576xbfloat16(%arg0: !torch.vtensor<[165,576],bf16>, %arg1: !torch.vtensor<[576,576],bf16>) -> !torch.vtensor<[165,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[165,576],bf16>, !torch.vtensor<[576,576],bf16> -> !torch.vtensor<[165,576],bf16>
    return %0 : !torch.vtensor<[165,576],bf16>
  }
}
