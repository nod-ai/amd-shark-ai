module {
  func.func public @fused_op_mm_f3f619a58d312ad39798112b0311e3323090b412_150000x16384xbfloat16_150000x4096xbfloat16(%arg0: !torch.vtensor<[150000,16384],bf16>, %arg1: !torch.vtensor<[150000,4096],bf16>) -> !torch.vtensor<[16384,4096],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[150000,16384],bf16>, !torch.list<int> -> !torch.vtensor<[16384,150000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[16384,150000],bf16>, !torch.vtensor<[150000,4096],bf16> -> !torch.vtensor<[16384,4096],bf16>
    return %2 : !torch.vtensor<[16384,4096],bf16>
  }
}
