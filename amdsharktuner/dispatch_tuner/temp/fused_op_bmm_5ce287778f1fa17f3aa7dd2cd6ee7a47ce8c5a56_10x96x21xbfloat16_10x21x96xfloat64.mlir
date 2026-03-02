module {
  func.func public @fused_op_bmm_5ce287778f1fa17f3aa7dd2cd6ee7a47ce8c5a56_10x96x21xbfloat16_10x21x96xfloat64(%arg0: !torch.vtensor<[10,96,21],bf16>, %arg1: !torch.vtensor<[10,21,96],f64>) -> !torch.vtensor<[10,96,96],f64> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[10,96,21],bf16>, !torch.vtensor<[10,21,96],f64> -> !torch.vtensor<[10,96,96],f64>
    return %0 : !torch.vtensor<[10,96,96],f64>
  }
}
