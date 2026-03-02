module {
  func.func public @fused_op_bmm_b9f10f0c0e974467491a6403f38cdbd91b7d5b31_16x384x192xbfloat16_16x384x192xbfloat16(%arg0: !torch.vtensor<[16,384,192],bf16>, %arg1: !torch.vtensor<[16,384,192],bf16>) -> !torch.vtensor<[16,384,384],bf16> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[16,384,192],bf16>, !torch.list<int> -> !torch.vtensor<[16,192,384],bf16>
    %2 = torch.aten.bmm %arg0, %1 : !torch.vtensor<[16,384,192],bf16>, !torch.vtensor<[16,192,384],bf16> -> !torch.vtensor<[16,384,384],bf16>
    return %2 : !torch.vtensor<[16,384,384],bf16>
  }
}
