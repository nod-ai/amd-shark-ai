module {
  func.func public @fused_op_mm_ac54384c133573bdff81cc039475d74b1551682c_24576x576xbfloat16_576x2048xbfloat16(%arg0: !torch.vtensor<[24576,576],bf16>, %arg1: !torch.vtensor<[576,2048],bf16>) -> !torch.vtensor<[24576,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[24576,576],bf16>, !torch.vtensor<[576,2048],bf16> -> !torch.vtensor<[24576,2048],bf16>
    return %0 : !torch.vtensor<[24576,2048],bf16>
  }
}
