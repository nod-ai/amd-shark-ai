module {
  func.func public @fused_op_mm_7e2ebc614da303507b969d3b05cbed175ba4b43a_10x576xbfloat16_576x576xbfloat16(%arg0: !torch.vtensor<[10,576],bf16>, %arg1: !torch.vtensor<[576,576],bf16>) -> !torch.vtensor<[10,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[10,576],bf16>, !torch.vtensor<[576,576],bf16> -> !torch.vtensor<[10,576],bf16>
    return %0 : !torch.vtensor<[10,576],bf16>
  }
}
