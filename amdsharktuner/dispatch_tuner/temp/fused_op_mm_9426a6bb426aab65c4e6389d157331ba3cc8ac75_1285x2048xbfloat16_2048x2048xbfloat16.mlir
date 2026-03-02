module {
  func.func public @fused_op_mm_9426a6bb426aab65c4e6389d157331ba3cc8ac75_1285x2048xbfloat16_2048x2048xbfloat16(%arg0: !torch.vtensor<[1285,2048],bf16>, %arg1: !torch.vtensor<[2048,2048],bf16>) -> !torch.vtensor<[1285,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1285,2048],bf16>, !torch.vtensor<[2048,2048],bf16> -> !torch.vtensor<[1285,2048],bf16>
    return %0 : !torch.vtensor<[1285,2048],bf16>
  }
}
