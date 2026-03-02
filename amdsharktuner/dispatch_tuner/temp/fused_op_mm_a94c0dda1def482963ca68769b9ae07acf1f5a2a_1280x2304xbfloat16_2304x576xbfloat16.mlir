module {
  func.func public @fused_op_mm_a94c0dda1def482963ca68769b9ae07acf1f5a2a_1280x2304xbfloat16_2304x576xbfloat16(%arg0: !torch.vtensor<[1280,2304],bf16>, %arg1: !torch.vtensor<[2304,576],bf16>) -> !torch.vtensor<[1280,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1280,2304],bf16>, !torch.vtensor<[2304,576],bf16> -> !torch.vtensor<[1280,576],bf16>
    return %0 : !torch.vtensor<[1280,576],bf16>
  }
}
