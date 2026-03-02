module {
  func.func public @fused_op_mm_fd17ce07f719d3a8a5874a3ee64716376a7968a4_1285x3840xbfloat16_2048x3840xbfloat16(%arg0: !torch.vtensor<[1285,3840],bf16>, %arg1: !torch.vtensor<[2048,3840],bf16>) -> !torch.vtensor<[1285,2048],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[2048,3840],bf16>, !torch.list<int> -> !torch.vtensor<[3840,2048],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[1285,3840],bf16>, !torch.vtensor<[3840,2048],bf16> -> !torch.vtensor<[1285,2048],bf16>
    return %2 : !torch.vtensor<[1285,2048],bf16>
  }
}
