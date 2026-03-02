module {
  func.func public @fused_op_mm_5902e7b3f2e5a4dfdc641807593e4ca6e591f540_4112x8192xbfloat16_8192x2048xbfloat16(%arg0: !torch.vtensor<[4112,8192],bf16>, %arg1: !torch.vtensor<[8192,2048],bf16>) -> !torch.vtensor<[4112,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4112,8192],bf16>, !torch.vtensor<[8192,2048],bf16> -> !torch.vtensor<[4112,2048],bf16>
    return %0 : !torch.vtensor<[4112,2048],bf16>
  }
}
