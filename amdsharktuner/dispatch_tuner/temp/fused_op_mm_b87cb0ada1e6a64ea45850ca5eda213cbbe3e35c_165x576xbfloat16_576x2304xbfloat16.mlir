module {
  func.func public @fused_op_mm_b87cb0ada1e6a64ea45850ca5eda213cbbe3e35c_165x576xbfloat16_576x2304xbfloat16(%arg0: !torch.vtensor<[165,576],bf16>, %arg1: !torch.vtensor<[576,2304],bf16>) -> !torch.vtensor<[165,2304],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[165,576],bf16>, !torch.vtensor<[576,2304],bf16> -> !torch.vtensor<[165,2304],bf16>
    return %0 : !torch.vtensor<[165,2304],bf16>
  }
}
