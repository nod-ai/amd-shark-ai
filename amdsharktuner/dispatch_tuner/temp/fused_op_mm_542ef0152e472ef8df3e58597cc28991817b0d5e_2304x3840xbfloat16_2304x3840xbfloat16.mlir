module {
  func.func public @fused_op_mm_542ef0152e472ef8df3e58597cc28991817b0d5e_2304x3840xbfloat16_2304x3840xbfloat16(%arg0: !torch.vtensor<[2304,3840],bf16>, %arg1: !torch.vtensor<[2304,3840],bf16>) -> !torch.vtensor<[3840,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[2304,3840],bf16>, !torch.list<int> -> !torch.vtensor<[3840,2304],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[3840,2304],bf16>, !torch.vtensor<[2304,3840],bf16> -> !torch.vtensor<[3840,3840],bf16>
    return %2 : !torch.vtensor<[3840,3840],bf16>
  }
}
