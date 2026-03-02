module {
  func.func public @fused_op_mm_6c3244b7ab8b444a2c8d6561764827054787f5de_4112x2048xbfloat16_2048x8192xbfloat16(%arg0: !torch.vtensor<[4112,2048],bf16>, %arg1: !torch.vtensor<[2048,8192],bf16>) -> !torch.vtensor<[4112,8192],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4112,2048],bf16>, !torch.vtensor<[2048,8192],bf16> -> !torch.vtensor<[4112,8192],bf16>
    return %0 : !torch.vtensor<[4112,8192],bf16>
  }
}
