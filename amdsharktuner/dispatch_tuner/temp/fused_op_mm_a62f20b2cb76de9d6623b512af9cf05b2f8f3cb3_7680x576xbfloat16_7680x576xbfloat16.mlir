module {
  func.func public @fused_op_mm_a62f20b2cb76de9d6623b512af9cf05b2f8f3cb3_7680x576xbfloat16_7680x576xbfloat16(%arg0: !torch.vtensor<[7680,576],bf16>, %arg1: !torch.vtensor<[7680,576],bf16>) -> !torch.vtensor<[576,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[7680,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,7680],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[576,7680],bf16>, !torch.vtensor<[7680,576],bf16> -> !torch.vtensor<[576,576],bf16>
    return %2 : !torch.vtensor<[576,576],bf16>
  }
}
