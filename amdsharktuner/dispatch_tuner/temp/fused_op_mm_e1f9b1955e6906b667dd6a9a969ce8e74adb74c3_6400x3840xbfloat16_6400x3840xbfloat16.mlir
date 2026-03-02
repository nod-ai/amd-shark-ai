module {
  func.func public @fused_op_mm_e1f9b1955e6906b667dd6a9a969ce8e74adb74c3_6400x3840xbfloat16_6400x3840xbfloat16(%arg0: !torch.vtensor<[6400,3840],bf16>, %arg1: !torch.vtensor<[6400,3840],bf16>) -> !torch.vtensor<[3840,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[6400,3840],bf16>, !torch.list<int> -> !torch.vtensor<[3840,6400],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[3840,6400],bf16>, !torch.vtensor<[6400,3840],bf16> -> !torch.vtensor<[3840,3840],bf16>
    return %2 : !torch.vtensor<[3840,3840],bf16>
  }
}
