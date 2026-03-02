module {
  func.func public @fused_op_mm_c60e7b2a3feb807ef63ae3571840cdadf1ba28b3_150000x8192xbfloat16_8192x2048xbfloat16(%arg0: !torch.vtensor<[150000,8192],bf16>, %arg1: !torch.vtensor<[8192,2048],bf16>) -> !torch.vtensor<[150000,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,8192],bf16>, !torch.vtensor<[8192,2048],bf16> -> !torch.vtensor<[150000,2048],bf16>
    return %0 : !torch.vtensor<[150000,2048],bf16>
  }
}
