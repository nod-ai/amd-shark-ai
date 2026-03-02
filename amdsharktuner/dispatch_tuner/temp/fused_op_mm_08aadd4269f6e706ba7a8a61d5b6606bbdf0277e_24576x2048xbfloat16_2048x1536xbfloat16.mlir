module {
  func.func public @fused_op_mm_08aadd4269f6e706ba7a8a61d5b6606bbdf0277e_24576x2048xbfloat16_2048x1536xbfloat16(%arg0: !torch.vtensor<[24576,2048],bf16>, %arg1: !torch.vtensor<[2048,1536],bf16>) -> !torch.vtensor<[24576,1536],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[24576,2048],bf16>, !torch.vtensor<[2048,1536],bf16> -> !torch.vtensor<[24576,1536],bf16>
    return %0 : !torch.vtensor<[24576,1536],bf16>
  }
}
