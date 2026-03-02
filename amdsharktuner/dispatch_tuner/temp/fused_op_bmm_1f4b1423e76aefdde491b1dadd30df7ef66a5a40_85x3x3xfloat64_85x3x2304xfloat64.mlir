module {
  func.func public @fused_op_bmm_1f4b1423e76aefdde491b1dadd30df7ef66a5a40_85x3x3xfloat64_85x3x2304xfloat64(%arg0: !torch.vtensor<[85,3,3],f64>, %arg1: !torch.vtensor<[85,3,2304],f64>) -> !torch.vtensor<[85,3,2304],f64> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[85,3,3],f64>, !torch.vtensor<[85,3,2304],f64> -> !torch.vtensor<[85,3,2304],f64>
    return %0 : !torch.vtensor<[85,3,2304],f64>
  }
}
