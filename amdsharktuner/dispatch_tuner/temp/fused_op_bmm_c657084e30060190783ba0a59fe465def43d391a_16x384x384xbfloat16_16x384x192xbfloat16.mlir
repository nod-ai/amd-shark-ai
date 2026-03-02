module {
  func.func public @fused_op_bmm_c657084e30060190783ba0a59fe465def43d391a_16x384x384xbfloat16_16x384x192xbfloat16(%arg0: !torch.vtensor<[16,384,384],bf16>, %arg1: !torch.vtensor<[16,384,192],bf16>) -> !torch.vtensor<[16,384,192],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[16,384,384],bf16>, !torch.vtensor<[16,384,192],bf16> -> !torch.vtensor<[16,384,192],bf16>
    return %0 : !torch.vtensor<[16,384,192],bf16>
  }
}
