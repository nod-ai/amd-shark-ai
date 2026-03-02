module {
  func.func public @fused_op_mm_95f201e40dbb32edb7d386d75c905bc459443ac2_3670016x128xbfloat16_3670016x128xbfloat16(%arg0: !torch.vtensor<[3670016,128],bf16>, %arg1: !torch.vtensor<[3670016,128],bf16>) -> !torch.vtensor<[128,128],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[3670016,128],bf16>, !torch.list<int> -> !torch.vtensor<[128,3670016],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[128,3670016],bf16>, !torch.vtensor<[3670016,128],bf16> -> !torch.vtensor<[128,128],bf16>
    return %2 : !torch.vtensor<[128,128],bf16>
  }
}
