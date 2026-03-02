module {
  func.func public @fused_op_mm_3f8570c26c7b1ba6c063697818a2e31543248657_10x576xbfloat16_576x2304xbfloat16(%arg0: !torch.vtensor<[10,576],bf16>, %arg1: !torch.vtensor<[576,2304],bf16>) -> !torch.vtensor<[10,2304],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[10,576],bf16>, !torch.vtensor<[576,2304],bf16> -> !torch.vtensor<[10,2304],bf16>
    return %0 : !torch.vtensor<[10,2304],bf16>
  }
}
