module {
  func.func public @fused_op_mm_95a9428b80ba9b8014e2977adeb19a8d3319b5cc_1280x2304xbfloat16_1280x576xbfloat16(%arg0: !torch.vtensor<[1280,2304],bf16>, %arg1: !torch.vtensor<[1280,576],bf16>) -> !torch.vtensor<[2304,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[1280,2304],bf16>, !torch.list<int> -> !torch.vtensor<[2304,1280],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2304,1280],bf16>, !torch.vtensor<[1280,576],bf16> -> !torch.vtensor<[2304,576],bf16>
    return %2 : !torch.vtensor<[2304,576],bf16>
  }
}
