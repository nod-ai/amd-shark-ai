module {
  func.func public @fused_op_bmm_6e681fd7acb32ed9dd3e3e9fe55620ea35b6a697_128x3x3xfloat64_128x3x2304xfloat64(%arg0: !torch.vtensor<[128,3,3],f64>, %arg1: !torch.vtensor<[128,3,2304],f64>) -> !torch.vtensor<[128,3,2304],f64> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[128,3,3],f64>, !torch.vtensor<[128,3,2304],f64> -> !torch.vtensor<[128,3,2304],f64>
    return %0 : !torch.vtensor<[128,3,2304],f64>
  }
}
