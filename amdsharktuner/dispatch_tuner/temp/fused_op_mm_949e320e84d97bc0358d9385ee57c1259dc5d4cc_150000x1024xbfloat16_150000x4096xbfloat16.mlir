module {
  func.func public @fused_op_mm_949e320e84d97bc0358d9385ee57c1259dc5d4cc_150000x1024xbfloat16_150000x4096xbfloat16(%arg0: !torch.vtensor<[150000,1024],bf16>, %arg1: !torch.vtensor<[150000,4096],bf16>) -> !torch.vtensor<[1024,4096],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[150000,1024],bf16>, !torch.list<int> -> !torch.vtensor<[1024,150000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[1024,150000],bf16>, !torch.vtensor<[150000,4096],bf16> -> !torch.vtensor<[1024,4096],bf16>
    return %2 : !torch.vtensor<[1024,4096],bf16>
  }
}
