module {
  func.func public @fused_op_mm_b2e2dbf4d801dba5db1934434c81eb3de7217016_1285x8192xbfloat16_8192x2048xbfloat16(%arg0: !torch.vtensor<[1285,8192],bf16>, %arg1: !torch.vtensor<[8192,2048],bf16>) -> !torch.vtensor<[1285,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1285,8192],bf16>, !torch.vtensor<[8192,2048],bf16> -> !torch.vtensor<[1285,2048],bf16>
    return %0 : !torch.vtensor<[1285,2048],bf16>
  }
}
