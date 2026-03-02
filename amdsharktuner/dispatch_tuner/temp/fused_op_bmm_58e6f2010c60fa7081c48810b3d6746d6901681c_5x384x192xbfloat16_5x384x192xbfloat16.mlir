module {
  func.func public @fused_op_bmm_58e6f2010c60fa7081c48810b3d6746d6901681c_5x384x192xbfloat16_5x384x192xbfloat16(%arg0: !torch.vtensor<[5,384,192],bf16>, %arg1: !torch.vtensor<[5,384,192],bf16>) -> !torch.vtensor<[5,384,384],bf16> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[5,384,192],bf16>, !torch.list<int> -> !torch.vtensor<[5,192,384],bf16>
    %2 = torch.aten.bmm %arg0, %1 : !torch.vtensor<[5,384,192],bf16>, !torch.vtensor<[5,192,384],bf16> -> !torch.vtensor<[5,384,384],bf16>
    return %2 : !torch.vtensor<[5,384,384],bf16>
  }
}
