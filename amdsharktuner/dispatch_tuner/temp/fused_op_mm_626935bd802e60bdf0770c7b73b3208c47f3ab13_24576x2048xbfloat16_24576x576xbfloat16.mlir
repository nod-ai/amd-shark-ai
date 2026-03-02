module {
  func.func public @fused_op_mm_626935bd802e60bdf0770c7b73b3208c47f3ab13_24576x2048xbfloat16_24576x576xbfloat16(%arg0: !torch.vtensor<[24576,2048],bf16>, %arg1: !torch.vtensor<[24576,576],bf16>) -> !torch.vtensor<[2048,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[24576,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,24576],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2048,24576],bf16>, !torch.vtensor<[24576,576],bf16> -> !torch.vtensor<[2048,576],bf16>
    return %2 : !torch.vtensor<[2048,576],bf16>
  }
}
