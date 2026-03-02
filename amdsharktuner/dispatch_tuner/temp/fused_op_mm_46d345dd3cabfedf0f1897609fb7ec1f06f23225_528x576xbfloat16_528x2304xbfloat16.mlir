module {
  func.func public @fused_op_mm_46d345dd3cabfedf0f1897609fb7ec1f06f23225_528x576xbfloat16_528x2304xbfloat16(%arg0: !torch.vtensor<[528,576],bf16>, %arg1: !torch.vtensor<[528,2304],bf16>) -> !torch.vtensor<[576,2304],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[528,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,528],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[576,528],bf16>, !torch.vtensor<[528,2304],bf16> -> !torch.vtensor<[576,2304],bf16>
    return %2 : !torch.vtensor<[576,2304],bf16>
  }
}
