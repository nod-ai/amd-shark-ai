module {
  func.func public @fused_op_mm_3f6a6d693d48f21d83b0ea568ef8a9f07d08e25a_1280x3840xbfloat16_7680x3840xbfloat16(%arg0: !torch.vtensor<[1280,3840],bf16>, %arg1: !torch.vtensor<[7680,3840],bf16>) -> !torch.vtensor<[1280,7680],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[7680,3840],bf16>, !torch.list<int> -> !torch.vtensor<[3840,7680],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[1280,3840],bf16>, !torch.vtensor<[3840,7680],bf16> -> !torch.vtensor<[1280,7680],bf16>
    return %2 : !torch.vtensor<[1280,7680],bf16>
  }
}
