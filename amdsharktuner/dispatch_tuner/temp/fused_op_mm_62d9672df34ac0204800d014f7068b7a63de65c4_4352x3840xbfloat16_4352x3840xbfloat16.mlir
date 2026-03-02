module {
  func.func public @fused_op_mm_62d9672df34ac0204800d014f7068b7a63de65c4_4352x3840xbfloat16_4352x3840xbfloat16(%arg0: !torch.vtensor<[4352,3840],bf16>, %arg1: !torch.vtensor<[4352,3840],bf16>) -> !torch.vtensor<[3840,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[4352,3840],bf16>, !torch.list<int> -> !torch.vtensor<[3840,4352],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[3840,4352],bf16>, !torch.vtensor<[4352,3840],bf16> -> !torch.vtensor<[3840,3840],bf16>
    return %2 : !torch.vtensor<[3840,3840],bf16>
  }
}
