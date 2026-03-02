module {
  func.func public @fused_op_mm_99dc5470e06d695c35386328396cbe0150cfcf66_5x1024xbfloat16_1024x1024xbfloat16(%arg0: !torch.vtensor<[5,1024],bf16>, %arg1: !torch.vtensor<[1024,1024],bf16>) -> !torch.vtensor<[5,1024],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[5,1024],bf16>, !torch.vtensor<[1024,1024],bf16> -> !torch.vtensor<[5,1024],bf16>
    return %0 : !torch.vtensor<[5,1024],bf16>
  }
}
