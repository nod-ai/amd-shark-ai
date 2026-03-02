module {
  func.func public @fused_op_mm_512bb641c803cb3732f90390091b7546a238b969_150000x4096xbfloat16_150000x16384xbfloat16(%arg0: !torch.vtensor<[150000,4096],bf16>, %arg1: !torch.vtensor<[150000,16384],bf16>) -> !torch.vtensor<[4096,16384],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[150000,4096],bf16>, !torch.list<int> -> !torch.vtensor<[4096,150000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[4096,150000],bf16>, !torch.vtensor<[150000,16384],bf16> -> !torch.vtensor<[4096,16384],bf16>
    return %2 : !torch.vtensor<[4096,16384],bf16>
  }
}
