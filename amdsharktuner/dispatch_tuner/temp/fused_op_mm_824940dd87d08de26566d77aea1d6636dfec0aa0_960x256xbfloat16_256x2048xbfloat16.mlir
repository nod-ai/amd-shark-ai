module {
  func.func public @fused_op_mm_824940dd87d08de26566d77aea1d6636dfec0aa0_960x256xbfloat16_256x2048xbfloat16(%arg0: !torch.vtensor<[960,256],bf16>, %arg1: !torch.vtensor<[256,2048],bf16>) -> !torch.vtensor<[960,2048],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[960,256],bf16>, !torch.vtensor<[256,2048],bf16> -> !torch.vtensor<[960,2048],bf16>
    return %0 : !torch.vtensor<[960,2048],bf16>
  }
}
