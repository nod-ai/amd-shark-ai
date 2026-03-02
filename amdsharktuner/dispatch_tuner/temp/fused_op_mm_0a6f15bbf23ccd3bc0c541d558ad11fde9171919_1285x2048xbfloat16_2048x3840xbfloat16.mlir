module {
  func.func public @fused_op_mm_0a6f15bbf23ccd3bc0c541d558ad11fde9171919_1285x2048xbfloat16_2048x3840xbfloat16(%arg0: !torch.vtensor<[1285,2048],bf16>, %arg1: !torch.vtensor<[2048,3840],bf16>) -> !torch.vtensor<[1285,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1285,2048],bf16>, !torch.vtensor<[2048,3840],bf16> -> !torch.vtensor<[1285,3840],bf16>
    return %0 : !torch.vtensor<[1285,3840],bf16>
  }
}
