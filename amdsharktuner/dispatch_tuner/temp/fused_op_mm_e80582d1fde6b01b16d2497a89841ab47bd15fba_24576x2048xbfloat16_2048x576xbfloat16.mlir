module {
  func.func public @fused_op_mm_e80582d1fde6b01b16d2497a89841ab47bd15fba_24576x2048xbfloat16_2048x576xbfloat16(%arg0: !torch.vtensor<[24576,2048],bf16>, %arg1: !torch.vtensor<[2048,576],bf16>) -> !torch.vtensor<[24576,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[24576,2048],bf16>, !torch.vtensor<[2048,576],bf16> -> !torch.vtensor<[24576,576],bf16>
    return %0 : !torch.vtensor<[24576,576],bf16>
  }
}
