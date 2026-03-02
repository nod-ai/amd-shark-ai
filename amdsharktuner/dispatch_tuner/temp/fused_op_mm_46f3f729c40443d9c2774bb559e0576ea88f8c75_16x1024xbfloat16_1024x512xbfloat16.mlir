module {
  func.func public @fused_op_mm_46f3f729c40443d9c2774bb559e0576ea88f8c75_16x1024xbfloat16_1024x512xbfloat16(%arg0: !torch.vtensor<[16,1024],bf16>, %arg1: !torch.vtensor<[1024,512],bf16>) -> !torch.vtensor<[16,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[16,1024],bf16>, !torch.vtensor<[1024,512],bf16> -> !torch.vtensor<[16,512],bf16>
    return %0 : !torch.vtensor<[16,512],bf16>
  }
}
