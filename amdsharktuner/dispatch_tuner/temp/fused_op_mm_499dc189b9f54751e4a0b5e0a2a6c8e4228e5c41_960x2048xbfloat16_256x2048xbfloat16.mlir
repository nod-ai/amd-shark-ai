module {
  func.func public @fused_op_mm_499dc189b9f54751e4a0b5e0a2a6c8e4228e5c41_960x2048xbfloat16_256x2048xbfloat16(%arg0: !torch.vtensor<[960,2048],bf16>, %arg1: !torch.vtensor<[256,2048],bf16>) -> !torch.vtensor<[960,256],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[256,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,256],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[960,2048],bf16>, !torch.vtensor<[2048,256],bf16> -> !torch.vtensor<[960,256],bf16>
    return %2 : !torch.vtensor<[960,256],bf16>
  }
}
