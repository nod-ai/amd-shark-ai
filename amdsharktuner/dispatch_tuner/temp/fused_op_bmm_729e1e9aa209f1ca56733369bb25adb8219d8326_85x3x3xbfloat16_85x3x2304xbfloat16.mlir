module {
  func.func public @fused_op_bmm_729e1e9aa209f1ca56733369bb25adb8219d8326_85x3x3xbfloat16_85x3x2304xbfloat16(%arg0: !torch.vtensor<[85,3,3],bf16>, %arg1: !torch.vtensor<[85,3,2304],bf16>) -> !torch.vtensor<[85,3,2304],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[85,3,3],bf16>, !torch.vtensor<[85,3,2304],bf16> -> !torch.vtensor<[85,3,2304],bf16>
    return %0 : !torch.vtensor<[85,3,2304],bf16>
  }
}
