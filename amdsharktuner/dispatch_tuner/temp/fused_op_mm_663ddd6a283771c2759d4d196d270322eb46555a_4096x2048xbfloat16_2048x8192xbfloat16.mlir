module {
  func.func public @fused_op_mm_663ddd6a283771c2759d4d196d270322eb46555a_4096x2048xbfloat16_2048x8192xbfloat16(%arg0: !torch.vtensor<[4096,2048],bf16>, %arg1: !torch.vtensor<[2048,8192],bf16>) -> !torch.vtensor<[4096,8192],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4096,2048],bf16>, !torch.vtensor<[2048,8192],bf16> -> !torch.vtensor<[4096,8192],bf16>
    return %0 : !torch.vtensor<[4096,8192],bf16>
  }
}
