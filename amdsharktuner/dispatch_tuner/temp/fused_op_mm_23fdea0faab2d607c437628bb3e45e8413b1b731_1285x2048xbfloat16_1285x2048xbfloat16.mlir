module {
  func.func public @fused_op_mm_23fdea0faab2d607c437628bb3e45e8413b1b731_1285x2048xbfloat16_1285x2048xbfloat16(%arg0: !torch.vtensor<[1285,2048],bf16>, %arg1: !torch.vtensor<[1285,2048],bf16>) -> !torch.vtensor<[2048,2048],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[1285,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,1285],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2048,1285],bf16>, !torch.vtensor<[1285,2048],bf16> -> !torch.vtensor<[2048,2048],bf16>
    return %2 : !torch.vtensor<[2048,2048],bf16>
  }
}
