module {
  func.func public @fused_op_mm_aed81e5394628775902ed8a4d6ad3ecb7cf7315f_150000x4096xbfloat16_150000x2048xbfloat16(%arg0: !torch.vtensor<[150000,4096],bf16>, %arg1: !torch.vtensor<[150000,2048],bf16>) -> !torch.vtensor<[4096,2048],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[150000,4096],bf16>, !torch.list<int> -> !torch.vtensor<[4096,150000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[4096,150000],bf16>, !torch.vtensor<[150000,2048],bf16> -> !torch.vtensor<[4096,2048],bf16>
    return %2 : !torch.vtensor<[4096,2048],bf16>
  }
}
