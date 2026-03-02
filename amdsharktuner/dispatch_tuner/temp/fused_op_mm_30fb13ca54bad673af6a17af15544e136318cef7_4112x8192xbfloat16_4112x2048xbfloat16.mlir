module {
  func.func public @fused_op_mm_30fb13ca54bad673af6a17af15544e136318cef7_4112x8192xbfloat16_4112x2048xbfloat16(%arg0: !torch.vtensor<[4112,8192],bf16>, %arg1: !torch.vtensor<[4112,2048],bf16>) -> !torch.vtensor<[8192,2048],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[4112,8192],bf16>, !torch.list<int> -> !torch.vtensor<[8192,4112],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[8192,4112],bf16>, !torch.vtensor<[4112,2048],bf16> -> !torch.vtensor<[8192,2048],bf16>
    return %2 : !torch.vtensor<[8192,2048],bf16>
  }
}
