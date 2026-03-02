module {
  func.func public @fused_op_mm_11568dfec6ed575bd4e00fd1d0f5f1310302d6ae_16x1024xbfloat16_1024x1024xbfloat16(%arg0: !torch.vtensor<[16,1024],bf16>, %arg1: !torch.vtensor<[1024,1024],bf16>) -> !torch.vtensor<[16,1024],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[16,1024],bf16>, !torch.vtensor<[1024,1024],bf16> -> !torch.vtensor<[16,1024],bf16>
    return %0 : !torch.vtensor<[16,1024],bf16>
  }
}
