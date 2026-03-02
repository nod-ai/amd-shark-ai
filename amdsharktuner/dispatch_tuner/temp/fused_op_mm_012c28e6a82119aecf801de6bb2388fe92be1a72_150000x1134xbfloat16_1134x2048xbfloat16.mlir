module {
  func.func public @fused_op_mm_012c28e6a82119aecf801de6bb2388fe92be1a72_150000x1134xbfloat16_1134x2048xbfloat16(%arg0: !torch.vtensor<[150000,1134],bf16>, %arg1: !torch.vtensor<[1134,2048],bf16>) -> !torch.vtensor<[150000,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,1134],bf16>, !torch.vtensor<[1134,2048],bf16> -> !torch.vtensor<[150000,2048],bf16>
    return %0 : !torch.vtensor<[150000,2048],bf16>
  }
}
