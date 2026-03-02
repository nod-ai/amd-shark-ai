module {
  func.func public @fused_op_mm_6d60ae81c21e05e510e9072ff5b06a70a0e61459_32x2304xbfloat16_2304x576xbfloat16(%arg0: !torch.vtensor<[32,2304],bf16>, %arg1: !torch.vtensor<[2304,576],bf16>) -> !torch.vtensor<[32,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[32,2304],bf16>, !torch.vtensor<[2304,576],bf16> -> !torch.vtensor<[32,576],bf16>
    return %0 : !torch.vtensor<[32,576],bf16>
  }
}
