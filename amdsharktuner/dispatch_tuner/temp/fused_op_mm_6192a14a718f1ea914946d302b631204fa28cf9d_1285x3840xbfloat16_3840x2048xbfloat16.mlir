module {
  func.func public @fused_op_mm_6192a14a718f1ea914946d302b631204fa28cf9d_1285x3840xbfloat16_3840x2048xbfloat16(%arg0: !torch.vtensor<[1285,3840],bf16>, %arg1: !torch.vtensor<[3840,2048],bf16>) -> !torch.vtensor<[1285,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1285,3840],bf16>, !torch.vtensor<[3840,2048],bf16> -> !torch.vtensor<[1285,2048],bf16>
    return %0 : !torch.vtensor<[1285,2048],bf16>
  }
}
