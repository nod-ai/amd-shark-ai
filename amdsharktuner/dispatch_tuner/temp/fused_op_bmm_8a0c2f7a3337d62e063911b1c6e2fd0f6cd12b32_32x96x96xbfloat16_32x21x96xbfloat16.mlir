module {
  func.func public @fused_op_bmm_8a0c2f7a3337d62e063911b1c6e2fd0f6cd12b32_32x96x96xbfloat16_32x21x96xbfloat16(%arg0: !torch.vtensor<[32,96,96],bf16>, %arg1: !torch.vtensor<[32,21,96],bf16>) -> !torch.vtensor<[32,96,21],bf16> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[32,21,96],bf16>, !torch.list<int> -> !torch.vtensor<[32,96,21],bf16>
    %2 = torch.aten.bmm %arg0, %1 : !torch.vtensor<[32,96,96],bf16>, !torch.vtensor<[32,96,21],bf16> -> !torch.vtensor<[32,96,21],bf16>
    return %2 : !torch.vtensor<[32,96,21],bf16>
  }
}
