module {
  func.func public @fused_op_mm_e12aa3f743df2e06ea021bd0d416dc8ba1fe8bb3_1280x576xbfloat16_1152x576xbfloat16(%arg0: !torch.vtensor<[1280,576],bf16>, %arg1: !torch.vtensor<[1152,576],bf16>) -> !torch.vtensor<[1280,1152],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[1152,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,1152],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[1280,576],bf16>, !torch.vtensor<[576,1152],bf16> -> !torch.vtensor<[1280,1152],bf16>
    return %2 : !torch.vtensor<[1280,1152],bf16>
  }
}
