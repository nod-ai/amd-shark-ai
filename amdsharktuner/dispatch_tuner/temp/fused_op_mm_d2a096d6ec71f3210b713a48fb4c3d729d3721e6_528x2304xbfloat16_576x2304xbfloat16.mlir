module {
  func.func public @fused_op_mm_d2a096d6ec71f3210b713a48fb4c3d729d3721e6_528x2304xbfloat16_576x2304xbfloat16(%arg0: !torch.vtensor<[528,2304],bf16>, %arg1: !torch.vtensor<[576,2304],bf16>) -> !torch.vtensor<[528,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[576,2304],bf16>, !torch.list<int> -> !torch.vtensor<[2304,576],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[528,2304],bf16>, !torch.vtensor<[2304,576],bf16> -> !torch.vtensor<[528,576],bf16>
    return %2 : !torch.vtensor<[528,576],bf16>
  }
}
