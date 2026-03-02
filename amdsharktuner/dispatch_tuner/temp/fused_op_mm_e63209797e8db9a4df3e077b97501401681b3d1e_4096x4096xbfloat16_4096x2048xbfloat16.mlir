module {
  func.func public @fused_op_mm_e63209797e8db9a4df3e077b97501401681b3d1e_4096x4096xbfloat16_4096x2048xbfloat16(%arg0: !torch.vtensor<[4096,4096],bf16>, %arg1: !torch.vtensor<[4096,2048],bf16>) -> !torch.vtensor<[4096,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4096,4096],bf16>, !torch.vtensor<[4096,2048],bf16> -> !torch.vtensor<[4096,2048],bf16>
    return %0 : !torch.vtensor<[4096,2048],bf16>
  }
}
