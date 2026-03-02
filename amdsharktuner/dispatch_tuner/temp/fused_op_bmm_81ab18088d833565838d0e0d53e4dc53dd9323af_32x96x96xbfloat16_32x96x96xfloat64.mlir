module {
  func.func public @fused_op_bmm_81ab18088d833565838d0e0d53e4dc53dd9323af_32x96x96xbfloat16_32x96x96xfloat64(%arg0: !torch.vtensor<[32,96,96],bf16>, %arg1: !torch.vtensor<[32,96,96],f64>) -> !torch.vtensor<[32,96,96],f64> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[32,96,96],bf16>, !torch.vtensor<[32,96,96],f64> -> !torch.vtensor<[32,96,96],f64>
    return %0 : !torch.vtensor<[32,96,96],f64>
  }
}
