module {
  func.func public @fused_op_mm_3d73bf58ee0769d1028dbc79275b963eaef9fa7a_1285x2048xbfloat16_8192x2048xbfloat16(%arg0: !torch.vtensor<[1285,2048],bf16>, %arg1: !torch.vtensor<[8192,2048],bf16>) -> !torch.vtensor<[1285,8192],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[8192,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,8192],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[1285,2048],bf16>, !torch.vtensor<[2048,8192],bf16> -> !torch.vtensor<[1285,8192],bf16>
    return %2 : !torch.vtensor<[1285,8192],bf16>
  }
}
