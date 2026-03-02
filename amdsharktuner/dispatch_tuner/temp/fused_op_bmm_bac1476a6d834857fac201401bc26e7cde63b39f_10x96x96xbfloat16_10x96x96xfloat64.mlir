module {
  func.func public @fused_op_bmm_bac1476a6d834857fac201401bc26e7cde63b39f_10x96x96xbfloat16_10x96x96xfloat64(%arg0: !torch.vtensor<[10,96,96],bf16>, %arg1: !torch.vtensor<[10,96,96],f64>) -> !torch.vtensor<[10,96,96],f64> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[10,96,96],bf16>, !torch.vtensor<[10,96,96],f64> -> !torch.vtensor<[10,96,96],f64>
    return %0 : !torch.vtensor<[10,96,96],f64>
  }
}
