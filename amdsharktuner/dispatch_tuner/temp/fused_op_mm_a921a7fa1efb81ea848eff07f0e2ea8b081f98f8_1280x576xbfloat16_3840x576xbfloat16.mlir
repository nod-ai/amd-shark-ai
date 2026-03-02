module {
  func.func public @fused_op_mm_a921a7fa1efb81ea848eff07f0e2ea8b081f98f8_1280x576xbfloat16_3840x576xbfloat16(%arg0: !torch.vtensor<[1280,576],bf16>, %arg1: !torch.vtensor<[3840,576],bf16>) -> !torch.vtensor<[1280,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[3840,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,3840],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[1280,576],bf16>, !torch.vtensor<[576,3840],bf16> -> !torch.vtensor<[1280,3840],bf16>
    return %2 : !torch.vtensor<[1280,3840],bf16>
  }
}
