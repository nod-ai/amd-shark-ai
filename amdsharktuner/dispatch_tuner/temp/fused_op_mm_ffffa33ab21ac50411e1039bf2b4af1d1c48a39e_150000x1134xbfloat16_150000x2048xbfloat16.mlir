module {
  func.func public @fused_op_mm_ffffa33ab21ac50411e1039bf2b4af1d1c48a39e_150000x1134xbfloat16_150000x2048xbfloat16(%arg0: !torch.vtensor<[150000,1134],bf16>, %arg1: !torch.vtensor<[150000,2048],bf16>) -> !torch.vtensor<[1134,2048],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[150000,1134],bf16>, !torch.list<int> -> !torch.vtensor<[1134,150000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[1134,150000],bf16>, !torch.vtensor<[150000,2048],bf16> -> !torch.vtensor<[1134,2048],bf16>
    return %2 : !torch.vtensor<[1134,2048],bf16>
  }
}
