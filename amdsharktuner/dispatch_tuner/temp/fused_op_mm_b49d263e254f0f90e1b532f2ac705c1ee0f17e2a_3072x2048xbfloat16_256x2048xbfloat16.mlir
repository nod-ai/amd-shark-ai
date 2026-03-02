module {
  func.func public @fused_op_mm_b49d263e254f0f90e1b532f2ac705c1ee0f17e2a_3072x2048xbfloat16_256x2048xbfloat16(%arg0: !torch.vtensor<[3072,2048],bf16>, %arg1: !torch.vtensor<[256,2048],bf16>) -> !torch.vtensor<[3072,256],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[256,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,256],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[3072,2048],bf16>, !torch.vtensor<[2048,256],bf16> -> !torch.vtensor<[3072,256],bf16>
    return %2 : !torch.vtensor<[3072,256],bf16>
  }
}
