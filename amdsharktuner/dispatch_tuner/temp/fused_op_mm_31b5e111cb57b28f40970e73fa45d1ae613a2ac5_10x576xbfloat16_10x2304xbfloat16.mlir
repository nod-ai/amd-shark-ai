module {
  func.func public @fused_op_mm_31b5e111cb57b28f40970e73fa45d1ae613a2ac5_10x576xbfloat16_10x2304xbfloat16(%arg0: !torch.vtensor<[10,576],bf16>, %arg1: !torch.vtensor<[10,2304],bf16>) -> !torch.vtensor<[576,2304],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[10,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,10],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[576,10],bf16>, !torch.vtensor<[10,2304],bf16> -> !torch.vtensor<[576,2304],bf16>
    return %2 : !torch.vtensor<[576,2304],bf16>
  }
}
