module {
  func.func public @fused_op_mm_d8c9e69d94b76efbb432d96469f048317ab97a5b_150000x2268xbfloat16_2268x4096xbfloat16(%arg0: !torch.vtensor<[150000,2268],bf16>, %arg1: !torch.vtensor<[2268,4096],bf16>) -> !torch.vtensor<[150000,4096],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,2268],bf16>, !torch.vtensor<[2268,4096],bf16> -> !torch.vtensor<[150000,4096],bf16>
    return %0 : !torch.vtensor<[150000,4096],bf16>
  }
}
