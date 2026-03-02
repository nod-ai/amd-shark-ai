module {
  func.func public @fused_op_mm_58f1d9c26ca724e44f878c93263a859376b340df_1285x2048xbfloat16_2048x8192xbfloat16(%arg0: !torch.vtensor<[1285,2048],bf16>, %arg1: !torch.vtensor<[2048,8192],bf16>) -> !torch.vtensor<[1285,8192],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1285,2048],bf16>, !torch.vtensor<[2048,8192],bf16> -> !torch.vtensor<[1285,8192],bf16>
    return %0 : !torch.vtensor<[1285,8192],bf16>
  }
}
