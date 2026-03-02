module {
  func.func public @fused_op_mm_a8e5260270684f1c7d279bfb7679430440b17b77_16800000x128xbfloat16_16800000x134xbfloat16(%arg0: !torch.vtensor<[16800000,128],bf16>, %arg1: !torch.vtensor<[16800000,134],bf16>) -> !torch.vtensor<[128,134],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[16800000,128],bf16>, !torch.list<int> -> !torch.vtensor<[128,16800000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[128,16800000],bf16>, !torch.vtensor<[16800000,134],bf16> -> !torch.vtensor<[128,134],bf16>
    return %2 : !torch.vtensor<[128,134],bf16>
  }
}
