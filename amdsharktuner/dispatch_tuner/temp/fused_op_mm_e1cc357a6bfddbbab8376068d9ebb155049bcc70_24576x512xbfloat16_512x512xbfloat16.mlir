module {
  func.func public @fused_op_mm_e1cc357a6bfddbbab8376068d9ebb155049bcc70_24576x512xbfloat16_512x512xbfloat16(%arg0: !torch.vtensor<[24576,512],bf16>, %arg1: !torch.vtensor<[512,512],bf16>) -> !torch.vtensor<[24576,512],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[24576,512],bf16>, !torch.vtensor<[512,512],bf16> -> !torch.vtensor<[24576,512],bf16>
    return %0 : !torch.vtensor<[24576,512],bf16>
  }
}
