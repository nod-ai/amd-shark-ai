module {
  func.func public @fused_op_mm_8bb12a641f159592f4397b94fa0a0966a13328f5_32x576xbfloat16_576x2304xbfloat16(%arg0: !torch.vtensor<[32,576],bf16>, %arg1: !torch.vtensor<[576,2304],bf16>) -> !torch.vtensor<[32,2304],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[32,576],bf16>, !torch.vtensor<[576,2304],bf16> -> !torch.vtensor<[32,2304],bf16>
    return %0 : !torch.vtensor<[32,2304],bf16>
  }
}
