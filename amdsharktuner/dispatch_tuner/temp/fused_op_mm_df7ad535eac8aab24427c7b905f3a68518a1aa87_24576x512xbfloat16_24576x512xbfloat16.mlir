module {
  func.func public @fused_op_mm_df7ad535eac8aab24427c7b905f3a68518a1aa87_24576x512xbfloat16_24576x512xbfloat16(%arg0: !torch.vtensor<[24576,512],bf16>, %arg1: !torch.vtensor<[24576,512],bf16>) -> !torch.vtensor<[512,512],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[24576,512],bf16>, !torch.list<int> -> !torch.vtensor<[512,24576],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[512,24576],bf16>, !torch.vtensor<[24576,512],bf16> -> !torch.vtensor<[512,512],bf16>
    return %2 : !torch.vtensor<[512,512],bf16>
  }
}
