module {
  func.func public @fused_op_mm_1cf3cd72f4c1e969938577d0a4c7fd56f371d7ce_1285x3840xbfloat16_1285x2048xbfloat16(%arg0: !torch.vtensor<[1285,3840],bf16>, %arg1: !torch.vtensor<[1285,2048],bf16>) -> !torch.vtensor<[3840,2048],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[1285,3840],bf16>, !torch.list<int> -> !torch.vtensor<[3840,1285],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[3840,1285],bf16>, !torch.vtensor<[1285,2048],bf16> -> !torch.vtensor<[3840,2048],bf16>
    return %2 : !torch.vtensor<[3840,2048],bf16>
  }
}
