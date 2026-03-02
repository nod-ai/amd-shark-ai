module {
  func.func public @fused_op_bmm_b3ae815a4656f94530c1aca3cd9175770f9d18a4_16x384x192xbfloat16_16x192x384xbfloat16(%arg0: !torch.vtensor<[16,384,192],bf16>, %arg1: !torch.vtensor<[16,192,384],bf16>) -> !torch.vtensor<[16,384,384],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[16,384,192],bf16>, !torch.vtensor<[16,192,384],bf16> -> !torch.vtensor<[16,384,384],bf16>
    return %0 : !torch.vtensor<[16,384,384],bf16>
  }
}
