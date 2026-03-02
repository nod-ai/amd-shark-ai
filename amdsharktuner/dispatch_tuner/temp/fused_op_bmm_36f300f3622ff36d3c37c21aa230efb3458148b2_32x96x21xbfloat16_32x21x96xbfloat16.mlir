module {
  func.func public @fused_op_bmm_36f300f3622ff36d3c37c21aa230efb3458148b2_32x96x21xbfloat16_32x21x96xbfloat16(%arg0: !torch.vtensor<[32,96,21],bf16>, %arg1: !torch.vtensor<[32,21,96],bf16>) -> !torch.vtensor<[32,96,96],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[32,96,21],bf16>, !torch.vtensor<[32,21,96],bf16> -> !torch.vtensor<[32,96,96],bf16>
    return %0 : !torch.vtensor<[32,96,96],bf16>
  }
}
