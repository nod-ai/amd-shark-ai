module {
  func.func public @fused_op_mm_16598e84d4ba63e7dce8fda98462355895c55dde_150000x1024xbfloat16_1024x128xbfloat16(%arg0: !torch.vtensor<[150000,1024],bf16>, %arg1: !torch.vtensor<[1024,128],bf16>) -> !torch.vtensor<[150000,128],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,1024],bf16>, !torch.vtensor<[1024,128],bf16> -> !torch.vtensor<[150000,128],bf16>
    return %0 : !torch.vtensor<[150000,128],bf16>
  }
}
