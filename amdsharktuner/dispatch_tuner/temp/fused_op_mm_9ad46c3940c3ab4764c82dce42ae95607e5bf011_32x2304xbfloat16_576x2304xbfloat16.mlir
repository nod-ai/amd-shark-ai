module {
  func.func public @fused_op_mm_9ad46c3940c3ab4764c82dce42ae95607e5bf011_32x2304xbfloat16_576x2304xbfloat16(%arg0: !torch.vtensor<[32,2304],bf16>, %arg1: !torch.vtensor<[576,2304],bf16>) -> !torch.vtensor<[32,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[576,2304],bf16>, !torch.list<int> -> !torch.vtensor<[2304,576],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[32,2304],bf16>, !torch.vtensor<[2304,576],bf16> -> !torch.vtensor<[32,576],bf16>
    return %2 : !torch.vtensor<[32,576],bf16>
  }
}
