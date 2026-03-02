module {
  func.func public @fused_op_mm_ba9e77bf554cd416a07cb7fe3916071bc4cb6495_150000x1024xbfloat16_3072x1024xbfloat16(%arg0: !torch.vtensor<[150000,1024],bf16>, %arg1: !torch.vtensor<[3072,1024],bf16>) -> !torch.vtensor<[150000,3072],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[3072,1024],bf16>, !torch.list<int> -> !torch.vtensor<[1024,3072],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[150000,1024],bf16>, !torch.vtensor<[1024,3072],bf16> -> !torch.vtensor<[150000,3072],bf16>
    return %2 : !torch.vtensor<[150000,3072],bf16>
  }
}
