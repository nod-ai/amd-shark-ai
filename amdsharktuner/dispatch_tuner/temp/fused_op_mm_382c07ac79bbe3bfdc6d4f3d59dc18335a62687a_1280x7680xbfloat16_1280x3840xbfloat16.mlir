module {
  func.func public @fused_op_mm_382c07ac79bbe3bfdc6d4f3d59dc18335a62687a_1280x7680xbfloat16_1280x3840xbfloat16(%arg0: !torch.vtensor<[1280,7680],bf16>, %arg1: !torch.vtensor<[1280,3840],bf16>) -> !torch.vtensor<[7680,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[1280,7680],bf16>, !torch.list<int> -> !torch.vtensor<[7680,1280],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[7680,1280],bf16>, !torch.vtensor<[1280,3840],bf16> -> !torch.vtensor<[7680,3840],bf16>
    return %2 : !torch.vtensor<[7680,3840],bf16>
  }
}
