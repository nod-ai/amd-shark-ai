module {
  func.func public @fused_op_mm_5bdb4cc5f642456c6165bf566a3641b89153194f_150000x2048xbfloat16_2048x8192xbfloat16(%arg0: !torch.vtensor<[150000,2048],bf16>, %arg1: !torch.vtensor<[2048,8192],bf16>) -> !torch.vtensor<[150000,8192],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,2048],bf16>, !torch.vtensor<[2048,8192],bf16> -> !torch.vtensor<[150000,8192],bf16>
    return %0 : !torch.vtensor<[150000,8192],bf16>
  }
}
