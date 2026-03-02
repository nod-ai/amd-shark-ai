module {
  func.func public @fused_op_mm_90bff78c5e44d6be249ca2f0fd08f18f842949b6_5x384xbfloat16_5x384xbfloat16(%arg0: !torch.vtensor<[5,384],bf16>, %arg1: !torch.vtensor<[5,384],bf16>) -> !torch.vtensor<[384,384],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[5,384],bf16>, !torch.list<int> -> !torch.vtensor<[384,5],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[384,5],bf16>, !torch.vtensor<[5,384],bf16> -> !torch.vtensor<[384,384],bf16>
    return %2 : !torch.vtensor<[384,384],bf16>
  }
}
