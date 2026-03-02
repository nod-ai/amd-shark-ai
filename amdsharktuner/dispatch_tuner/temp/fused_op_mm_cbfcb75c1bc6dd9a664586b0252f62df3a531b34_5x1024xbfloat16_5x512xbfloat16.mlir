module {
  func.func public @fused_op_mm_cbfcb75c1bc6dd9a664586b0252f62df3a531b34_5x1024xbfloat16_5x512xbfloat16(%arg0: !torch.vtensor<[5,1024],bf16>, %arg1: !torch.vtensor<[5,512],bf16>) -> !torch.vtensor<[1024,512],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[5,1024],bf16>, !torch.list<int> -> !torch.vtensor<[1024,5],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[1024,5],bf16>, !torch.vtensor<[5,512],bf16> -> !torch.vtensor<[1024,512],bf16>
    return %2 : !torch.vtensor<[1024,512],bf16>
  }
}
