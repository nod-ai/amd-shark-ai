module {
  func.func public @fused_op_mm_5d7f572ad1c3c5e43214bae931f5a21b7c15f0ea_5x1024xbfloat16_1024x512xbfloat16(%arg0: !torch.vtensor<[5,1024],bf16>, %arg1: !torch.vtensor<[1024,512],bf16>) -> !torch.vtensor<[5,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[5,1024],bf16>, !torch.vtensor<[1024,512],bf16> -> !torch.vtensor<[5,512],bf16>
    return %0 : !torch.vtensor<[5,512],bf16>
  }
}
