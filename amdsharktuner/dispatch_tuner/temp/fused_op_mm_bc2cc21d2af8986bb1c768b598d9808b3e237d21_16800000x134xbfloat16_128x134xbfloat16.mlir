module {
  func.func public @fused_op_mm_bc2cc21d2af8986bb1c768b598d9808b3e237d21_16800000x134xbfloat16_128x134xbfloat16(%arg0: !torch.vtensor<[16800000,134],bf16>, %arg1: !torch.vtensor<[128,134],bf16>) -> !torch.vtensor<[16800000,128],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[128,134],bf16>, !torch.list<int> -> !torch.vtensor<[134,128],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[16800000,134],bf16>, !torch.vtensor<[134,128],bf16> -> !torch.vtensor<[16800000,128],bf16>
    return %2 : !torch.vtensor<[16800000,128],bf16>
  }
}
