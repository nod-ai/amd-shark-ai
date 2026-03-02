module {
  func.func public @fused_op_mm_a625c4a8af3eca01620d6a88c68c94b82b683bd1_7680x304xbfloat16_7680x512xbfloat16(%arg0: !torch.vtensor<[7680,304],bf16>, %arg1: !torch.vtensor<[7680,512],bf16>) -> !torch.vtensor<[304,512],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[7680,304],bf16>, !torch.list<int> -> !torch.vtensor<[304,7680],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[304,7680],bf16>, !torch.vtensor<[7680,512],bf16> -> !torch.vtensor<[304,512],bf16>
    return %2 : !torch.vtensor<[304,512],bf16>
  }
}
