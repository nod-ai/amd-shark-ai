module {
  func.func public @fused_op_mm_fe80376d094b7c3d7b46970e9bbe5396ce7cfc5f_150000x1024xbfloat16_1024x1024xbfloat16(%arg0: !torch.vtensor<[150000,1024],bf16>, %arg1: !torch.vtensor<[1024,1024],bf16>) -> !torch.vtensor<[150000,1024],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[1024,1024],bf16>, !torch.list<int> -> !torch.vtensor<[1024,1024],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[150000,1024],bf16>, !torch.vtensor<[1024,1024],bf16> -> !torch.vtensor<[150000,1024],bf16>
    return %2 : !torch.vtensor<[150000,1024],bf16>
  }
}
