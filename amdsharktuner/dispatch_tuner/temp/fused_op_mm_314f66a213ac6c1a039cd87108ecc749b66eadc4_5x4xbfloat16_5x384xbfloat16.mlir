module {
  func.func public @fused_op_mm_314f66a213ac6c1a039cd87108ecc749b66eadc4_5x4xbfloat16_5x384xbfloat16(%arg0: !torch.vtensor<[5,4],bf16>, %arg1: !torch.vtensor<[5,384],bf16>) -> !torch.vtensor<[4,384],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[5,4],bf16>, !torch.list<int> -> !torch.vtensor<[4,5],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[4,5],bf16>, !torch.vtensor<[5,384],bf16> -> !torch.vtensor<[4,384],bf16>
    return %2 : !torch.vtensor<[4,384],bf16>
  }
}
