module {
  func.func public @fused_op_mm_30ef1e0210111b73d9e49dcf4b77927e7646edd6_1280x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[1280,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[1280,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[3840,3840],bf16>, !torch.list<int> -> !torch.vtensor<[3840,3840],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[1280,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[1280,3840],bf16>
    return %2 : !torch.vtensor<[1280,3840],bf16>
  }
}
