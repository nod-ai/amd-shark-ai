module {
  func.func public @fused_op_mm_4987af4b30674002fb4ef98a53aa364461746f54_150000x3072xbfloat16_150000x1024xbfloat16(%arg0: !torch.vtensor<[150000,3072],bf16>, %arg1: !torch.vtensor<[150000,1024],bf16>) -> !torch.vtensor<[3072,1024],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[150000,3072],bf16>, !torch.list<int> -> !torch.vtensor<[3072,150000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[3072,150000],bf16>, !torch.vtensor<[150000,1024],bf16> -> !torch.vtensor<[3072,1024],bf16>
    return %2 : !torch.vtensor<[3072,1024],bf16>
  }
}
