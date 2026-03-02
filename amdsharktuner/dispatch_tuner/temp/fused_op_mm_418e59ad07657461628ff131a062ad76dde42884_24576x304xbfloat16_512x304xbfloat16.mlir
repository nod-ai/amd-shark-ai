module {
  func.func public @fused_op_mm_418e59ad07657461628ff131a062ad76dde42884_24576x304xbfloat16_512x304xbfloat16(%arg0: !torch.vtensor<[24576,304],bf16>, %arg1: !torch.vtensor<[512,304],bf16>) -> !torch.vtensor<[24576,512],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[512,304],bf16>, !torch.list<int> -> !torch.vtensor<[304,512],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[24576,304],bf16>, !torch.vtensor<[304,512],bf16> -> !torch.vtensor<[24576,512],bf16>
    return %2 : !torch.vtensor<[24576,512],bf16>
  }
}
