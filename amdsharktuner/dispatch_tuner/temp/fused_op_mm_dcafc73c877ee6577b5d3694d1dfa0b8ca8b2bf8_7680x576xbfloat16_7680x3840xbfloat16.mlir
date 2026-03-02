module {
  func.func public @fused_op_mm_dcafc73c877ee6577b5d3694d1dfa0b8ca8b2bf8_7680x576xbfloat16_7680x3840xbfloat16(%arg0: !torch.vtensor<[7680,576],bf16>, %arg1: !torch.vtensor<[7680,3840],bf16>) -> !torch.vtensor<[576,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[7680,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,7680],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[576,7680],bf16>, !torch.vtensor<[7680,3840],bf16> -> !torch.vtensor<[576,3840],bf16>
    return %2 : !torch.vtensor<[576,3840],bf16>
  }
}
