module {
  func.func public @fused_op_mm_5162f186214cd313994c3fd4ab6fc791ddb3dfa0_3072x256xbfloat16_3072x2048xbfloat16(%arg0: !torch.vtensor<[3072,256],bf16>, %arg1: !torch.vtensor<[3072,2048],bf16>) -> !torch.vtensor<[256,2048],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[3072,256],bf16>, !torch.list<int> -> !torch.vtensor<[256,3072],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[256,3072],bf16>, !torch.vtensor<[3072,2048],bf16> -> !torch.vtensor<[256,2048],bf16>
    return %2 : !torch.vtensor<[256,2048],bf16>
  }
}
