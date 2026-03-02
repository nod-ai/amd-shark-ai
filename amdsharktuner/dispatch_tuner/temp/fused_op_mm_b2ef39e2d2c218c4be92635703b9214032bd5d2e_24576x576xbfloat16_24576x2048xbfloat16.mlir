module {
  func.func public @fused_op_mm_b2ef39e2d2c218c4be92635703b9214032bd5d2e_24576x576xbfloat16_24576x2048xbfloat16(%arg0: !torch.vtensor<[24576,576],bf16>, %arg1: !torch.vtensor<[24576,2048],bf16>) -> !torch.vtensor<[576,2048],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[24576,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,24576],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[576,24576],bf16>, !torch.vtensor<[24576,2048],bf16> -> !torch.vtensor<[576,2048],bf16>
    return %2 : !torch.vtensor<[576,2048],bf16>
  }
}
