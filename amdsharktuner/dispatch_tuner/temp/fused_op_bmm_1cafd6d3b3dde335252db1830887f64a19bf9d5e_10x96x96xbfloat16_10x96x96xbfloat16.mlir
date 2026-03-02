module {
  func.func public @fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16(%arg0: !torch.vtensor<[10,96,96],bf16>, %arg1: !torch.vtensor<[10,96,96],bf16>) -> !torch.vtensor<[10,96,96],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[10,96,96],bf16>, !torch.vtensor<[10,96,96],bf16> -> !torch.vtensor<[10,96,96],bf16>
    return %0 : !torch.vtensor<[10,96,96],bf16>
  }
}
