module {
  func.func public @fused_op_mm_b097391c0044d405a93e0aba5648fe7736fcaff6_32x576xbfloat16_32x576xbfloat16(%arg0: !torch.vtensor<[32,576],bf16>, %arg1: !torch.vtensor<[32,576],bf16>) -> !torch.vtensor<[576,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[32,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,32],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[576,32],bf16>, !torch.vtensor<[32,576],bf16> -> !torch.vtensor<[576,576],bf16>
    return %2 : !torch.vtensor<[576,576],bf16>
  }
}
