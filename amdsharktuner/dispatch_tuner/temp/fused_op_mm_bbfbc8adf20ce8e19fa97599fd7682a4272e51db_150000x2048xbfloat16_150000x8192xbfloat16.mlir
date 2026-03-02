module {
  func.func public @fused_op_mm_bbfbc8adf20ce8e19fa97599fd7682a4272e51db_150000x2048xbfloat16_150000x8192xbfloat16(%arg0: !torch.vtensor<[150000,2048],bf16>, %arg1: !torch.vtensor<[150000,8192],bf16>) -> !torch.vtensor<[2048,8192],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[150000,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,150000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2048,150000],bf16>, !torch.vtensor<[150000,8192],bf16> -> !torch.vtensor<[2048,8192],bf16>
    return %2 : !torch.vtensor<[2048,8192],bf16>
  }
}
