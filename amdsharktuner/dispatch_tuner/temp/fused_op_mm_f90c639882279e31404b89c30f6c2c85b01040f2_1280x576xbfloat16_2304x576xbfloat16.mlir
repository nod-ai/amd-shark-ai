module {
  func.func public @fused_op_mm_f90c639882279e31404b89c30f6c2c85b01040f2_1280x576xbfloat16_2304x576xbfloat16(%arg0: !torch.vtensor<[1280,576],bf16>, %arg1: !torch.vtensor<[2304,576],bf16>) -> !torch.vtensor<[1280,2304],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[2304,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,2304],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[1280,576],bf16>, !torch.vtensor<[576,2304],bf16> -> !torch.vtensor<[1280,2304],bf16>
    return %2 : !torch.vtensor<[1280,2304],bf16>
  }
}
