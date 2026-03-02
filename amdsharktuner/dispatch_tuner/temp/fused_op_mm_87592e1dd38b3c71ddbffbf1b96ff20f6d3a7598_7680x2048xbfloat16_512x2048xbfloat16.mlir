module {
  func.func public @fused_op_mm_87592e1dd38b3c71ddbffbf1b96ff20f6d3a7598_7680x2048xbfloat16_512x2048xbfloat16(%arg0: !torch.vtensor<[7680,2048],bf16>, %arg1: !torch.vtensor<[512,2048],bf16>) -> !torch.vtensor<[7680,512],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[512,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,512],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[7680,2048],bf16>, !torch.vtensor<[2048,512],bf16> -> !torch.vtensor<[7680,512],bf16>
    return %2 : !torch.vtensor<[7680,512],bf16>
  }
}
