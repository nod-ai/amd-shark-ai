module {
  func.func public @fused_op_mm_ccc6043502d72e6e179ae1b1501d4eb8ffcb71bb_11520x3840xbfloat16_3840x3840xbfloat16(%arg0: !torch.vtensor<[11520,3840],bf16>, %arg1: !torch.vtensor<[3840,3840],bf16>) -> !torch.vtensor<[11520,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[3840,3840],bf16>, !torch.list<int> -> !torch.vtensor<[3840,3840],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[11520,3840],bf16>, !torch.vtensor<[3840,3840],bf16> -> !torch.vtensor<[11520,3840],bf16>
    return %2 : !torch.vtensor<[11520,3840],bf16>
  }
}
