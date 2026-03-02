module {
  func.func public @fused_op_mm_41dea93faee482880d27d1b40347da40a7ef3c5d_165x2304xbfloat16_2304x576xbfloat16(%arg0: !torch.vtensor<[165,2304],bf16>, %arg1: !torch.vtensor<[2304,576],bf16>) -> !torch.vtensor<[165,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[165,2304],bf16>, !torch.vtensor<[2304,576],bf16> -> !torch.vtensor<[165,576],bf16>
    return %0 : !torch.vtensor<[165,576],bf16>
  }
}
