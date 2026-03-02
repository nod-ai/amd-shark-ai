module {
  func.func public @fused_op_bmm_f3c8db517e88eb479a9fdd96348cdca14599dd89_32x96x96xbfloat16_32x96x96xbfloat16(%arg0: !torch.vtensor<[32,96,96],bf16>, %arg1: !torch.vtensor<[32,96,96],bf16>) -> !torch.vtensor<[32,96,96],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[32,96,96],bf16>, !torch.vtensor<[32,96,96],bf16> -> !torch.vtensor<[32,96,96],bf16>
    return %0 : !torch.vtensor<[32,96,96],bf16>
  }
}
