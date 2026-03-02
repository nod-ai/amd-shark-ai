module {
  func.func public @fused_op_mm_e7f9a2d4de82838b430371a620f0479154ba04d7_21760x20xbfloat16_21760x3840xbfloat16(%arg0: !torch.vtensor<[21760,20],bf16>, %arg1: !torch.vtensor<[21760,3840],bf16>) -> !torch.vtensor<[20,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[21760,20],bf16>, !torch.list<int> -> !torch.vtensor<[20,21760],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[20,21760],bf16>, !torch.vtensor<[21760,3840],bf16> -> !torch.vtensor<[20,3840],bf16>
    return %2 : !torch.vtensor<[20,3840],bf16>
  }
}
