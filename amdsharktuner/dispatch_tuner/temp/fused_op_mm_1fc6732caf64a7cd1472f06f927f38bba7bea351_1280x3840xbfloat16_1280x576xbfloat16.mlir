module {
  func.func public @fused_op_mm_1fc6732caf64a7cd1472f06f927f38bba7bea351_1280x3840xbfloat16_1280x576xbfloat16(%arg0: !torch.vtensor<[1280,3840],bf16>, %arg1: !torch.vtensor<[1280,576],bf16>) -> !torch.vtensor<[3840,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[1280,3840],bf16>, !torch.list<int> -> !torch.vtensor<[3840,1280],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[3840,1280],bf16>, !torch.vtensor<[1280,576],bf16> -> !torch.vtensor<[3840,576],bf16>
    return %2 : !torch.vtensor<[3840,576],bf16>
  }
}
