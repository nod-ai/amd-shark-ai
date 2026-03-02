module {
  func.func public @fused_op_mm_0bd99756c6a3f296f5dccd50671b4146834e1a35_1280x576xbfloat16_1280x2304xbfloat16(%arg0: !torch.vtensor<[1280,576],bf16>, %arg1: !torch.vtensor<[1280,2304],bf16>) -> !torch.vtensor<[576,2304],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[1280,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,1280],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[576,1280],bf16>, !torch.vtensor<[1280,2304],bf16> -> !torch.vtensor<[576,2304],bf16>
    return %2 : !torch.vtensor<[576,2304],bf16>
  }
}
