module {
  func.func public @fused_op_mm_3ddb4bce77cd7db5615f375aa85bc3683f9eebde_2119936x128xbfloat16_128x128xbfloat16(%arg0: !torch.vtensor<[2119936,128],bf16>, %arg1: !torch.vtensor<[128,128],bf16>) -> !torch.vtensor<[2119936,128],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2119936,128],bf16>, !torch.vtensor<[128,128],bf16> -> !torch.vtensor<[2119936,128],bf16>
    return %0 : !torch.vtensor<[2119936,128],bf16>
  }
}
