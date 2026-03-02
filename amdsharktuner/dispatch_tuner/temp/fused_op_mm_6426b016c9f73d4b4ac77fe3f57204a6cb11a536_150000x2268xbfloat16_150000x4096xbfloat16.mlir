module {
  func.func public @fused_op_mm_6426b016c9f73d4b4ac77fe3f57204a6cb11a536_150000x2268xbfloat16_150000x4096xbfloat16(%arg0: !torch.vtensor<[150000,2268],bf16>, %arg1: !torch.vtensor<[150000,4096],bf16>) -> !torch.vtensor<[2268,4096],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[150000,2268],bf16>, !torch.list<int> -> !torch.vtensor<[2268,150000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2268,150000],bf16>, !torch.vtensor<[150000,4096],bf16> -> !torch.vtensor<[2268,4096],bf16>
    return %2 : !torch.vtensor<[2268,4096],bf16>
  }
}
