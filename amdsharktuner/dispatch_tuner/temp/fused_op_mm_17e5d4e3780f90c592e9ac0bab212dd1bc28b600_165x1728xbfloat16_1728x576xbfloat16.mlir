module {
  func.func public @fused_op_mm_17e5d4e3780f90c592e9ac0bab212dd1bc28b600_165x1728xbfloat16_1728x576xbfloat16(%arg0: !torch.vtensor<[165,1728],bf16>, %arg1: !torch.vtensor<[1728,576],bf16>) -> !torch.vtensor<[165,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[165,1728],bf16>, !torch.vtensor<[1728,576],bf16> -> !torch.vtensor<[165,576],bf16>
    return %0 : !torch.vtensor<[165,576],bf16>
  }
}
