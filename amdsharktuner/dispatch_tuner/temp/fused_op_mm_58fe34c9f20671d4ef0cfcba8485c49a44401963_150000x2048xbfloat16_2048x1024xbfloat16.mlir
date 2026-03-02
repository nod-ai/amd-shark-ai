module {
  func.func public @fused_op_mm_58fe34c9f20671d4ef0cfcba8485c49a44401963_150000x2048xbfloat16_2048x1024xbfloat16(%arg0: !torch.vtensor<[150000,2048],bf16>, %arg1: !torch.vtensor<[2048,1024],bf16>) -> !torch.vtensor<[150000,1024],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,2048],bf16>, !torch.vtensor<[2048,1024],bf16> -> !torch.vtensor<[150000,1024],bf16>
    return %0 : !torch.vtensor<[150000,1024],bf16>
  }
}
