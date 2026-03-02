module {
  func.func public @fused_op_mm_61526b2842a4ce8bcb032dd1bdb8cfefcf663053_4112x2048xbfloat16_4112x8192xbfloat16(%arg0: !torch.vtensor<[4112,2048],bf16>, %arg1: !torch.vtensor<[4112,8192],bf16>) -> !torch.vtensor<[2048,8192],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[4112,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,4112],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2048,4112],bf16>, !torch.vtensor<[4112,8192],bf16> -> !torch.vtensor<[2048,8192],bf16>
    return %2 : !torch.vtensor<[2048,8192],bf16>
  }
}
