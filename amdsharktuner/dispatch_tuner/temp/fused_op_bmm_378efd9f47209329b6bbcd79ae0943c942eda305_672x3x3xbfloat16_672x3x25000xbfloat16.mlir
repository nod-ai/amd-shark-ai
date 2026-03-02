module {
  func.func public @fused_op_bmm_378efd9f47209329b6bbcd79ae0943c942eda305_672x3x3xbfloat16_672x3x25000xbfloat16(%arg0: !torch.vtensor<[672,3,3],bf16>, %arg1: !torch.vtensor<[672,3,25000],bf16>) -> !torch.vtensor<[672,3,25000],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[672,3,3],bf16>, !torch.vtensor<[672,3,25000],bf16> -> !torch.vtensor<[672,3,25000],bf16>
    return %0 : !torch.vtensor<[672,3,25000],bf16>
  }
}
