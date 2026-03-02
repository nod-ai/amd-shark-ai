module {
  func.func public @fused_op_bmm_867027adba2210eaaafe9656e1c0693cb4700389_16x384x192xbfloat16_16x384x384xbfloat16(%arg0: !torch.vtensor<[16,384,192],bf16>, %arg1: !torch.vtensor<[16,384,384],bf16>) -> !torch.vtensor<[16,192,384],bf16> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[16,384,192],bf16>, !torch.list<int> -> !torch.vtensor<[16,192,384],bf16>
    %int0_0 = torch.constant.int 0
    %int2_1 = torch.constant.int 2
    %int1_2 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int0_0, %int2_1, %int1_2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.permute %arg1, %2 : !torch.vtensor<[16,384,384],bf16>, !torch.list<int> -> !torch.vtensor<[16,384,384],bf16>
    %4 = torch.aten.bmm %1, %3 : !torch.vtensor<[16,192,384],bf16>, !torch.vtensor<[16,384,384],bf16> -> !torch.vtensor<[16,192,384],bf16>
    return %4 : !torch.vtensor<[16,192,384],bf16>
  }
}
