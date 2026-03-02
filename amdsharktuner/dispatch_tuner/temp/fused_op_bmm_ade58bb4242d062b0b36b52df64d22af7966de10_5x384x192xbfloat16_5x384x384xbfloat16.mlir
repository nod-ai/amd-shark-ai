module {
  func.func public @fused_op_bmm_ade58bb4242d062b0b36b52df64d22af7966de10_5x384x192xbfloat16_5x384x384xbfloat16(%arg0: !torch.vtensor<[5,384,192],bf16>, %arg1: !torch.vtensor<[5,384,384],bf16>) -> !torch.vtensor<[5,192,384],bf16> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[5,384,192],bf16>, !torch.list<int> -> !torch.vtensor<[5,192,384],bf16>
    %int0_0 = torch.constant.int 0
    %int2_1 = torch.constant.int 2
    %int1_2 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int0_0, %int2_1, %int1_2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.permute %arg1, %2 : !torch.vtensor<[5,384,384],bf16>, !torch.list<int> -> !torch.vtensor<[5,384,384],bf16>
    %4 = torch.aten.bmm %1, %3 : !torch.vtensor<[5,192,384],bf16>, !torch.vtensor<[5,384,384],bf16> -> !torch.vtensor<[5,192,384],bf16>
    return %4 : !torch.vtensor<[5,192,384],bf16>
  }
}
