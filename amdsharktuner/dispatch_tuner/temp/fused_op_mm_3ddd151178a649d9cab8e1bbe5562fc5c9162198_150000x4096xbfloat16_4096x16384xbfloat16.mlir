module {
  func.func public @fused_op_mm_3ddd151178a649d9cab8e1bbe5562fc5c9162198_150000x4096xbfloat16_4096x16384xbfloat16(%arg0: !torch.vtensor<[150000,4096],bf16>, %arg1: !torch.vtensor<[4096,16384],bf16>) -> !torch.vtensor<[150000,16384],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,4096],bf16>, !torch.vtensor<[4096,16384],bf16> -> !torch.vtensor<[150000,16384],bf16>
    return %0 : !torch.vtensor<[150000,16384],bf16>
  }
}
