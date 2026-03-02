module {
  func.func public @fused_op_bmm_70dea66c9f7a1534f8fc8ed293ed33c5629a11ff_16x384x192xbfloat16_16x384x384xbfloat16(%arg0: !torch.vtensor<[16,384,192],bf16>, %arg1: !torch.vtensor<[16,384,384],bf16>) -> !torch.vtensor<[16,192,384],bf16> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[16,384,192],bf16>, !torch.list<int> -> !torch.vtensor<[16,192,384],bf16>
    %2 = torch.aten.bmm %1, %arg1 : !torch.vtensor<[16,192,384],bf16>, !torch.vtensor<[16,384,384],bf16> -> !torch.vtensor<[16,192,384],bf16>
    return %2 : !torch.vtensor<[16,192,384],bf16>
  }
}
