module {
  func.func public @fused_op_bmm_5e377616675d92553739036a8770183898458c26_32x96x21xbfloat16_32x21x96xfloat64(%arg0: !torch.vtensor<[32,96,21],bf16>, %arg1: !torch.vtensor<[32,21,96],f64>) -> !torch.vtensor<[32,96,96],f64> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[32,96,21],bf16>, !torch.vtensor<[32,21,96],f64> -> !torch.vtensor<[32,96,96],f64>
    return %0 : !torch.vtensor<[32,96,96],f64>
  }
}
