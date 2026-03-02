module {
  func.func public @fused_op_mm_32d852a128ee2bbc5997605219d7769ca01db2f8_150000x1024xbfloat16_150000x128xbfloat16(%arg0: !torch.vtensor<[150000,1024],bf16>, %arg1: !torch.vtensor<[150000,128],bf16>) -> !torch.vtensor<[1024,128],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[150000,1024],bf16>, !torch.list<int> -> !torch.vtensor<[1024,150000],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[1024,150000],bf16>, !torch.vtensor<[150000,128],bf16> -> !torch.vtensor<[1024,128],bf16>
    return %2 : !torch.vtensor<[1024,128],bf16>
  }
}
