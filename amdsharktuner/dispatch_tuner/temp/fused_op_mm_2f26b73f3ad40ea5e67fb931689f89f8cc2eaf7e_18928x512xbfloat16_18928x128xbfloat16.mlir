module {
  func.func public @fused_op_mm_2f26b73f3ad40ea5e67fb931689f89f8cc2eaf7e_18928x512xbfloat16_18928x128xbfloat16(%arg0: !torch.vtensor<[18928,512],bf16>, %arg1: !torch.vtensor<[18928,128],bf16>) -> !torch.vtensor<[512,128],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[18928,512],bf16>, !torch.list<int> -> !torch.vtensor<[512,18928],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[512,18928],bf16>, !torch.vtensor<[18928,128],bf16> -> !torch.vtensor<[512,128],bf16>
    return %2 : !torch.vtensor<[512,128],bf16>
  }
}
