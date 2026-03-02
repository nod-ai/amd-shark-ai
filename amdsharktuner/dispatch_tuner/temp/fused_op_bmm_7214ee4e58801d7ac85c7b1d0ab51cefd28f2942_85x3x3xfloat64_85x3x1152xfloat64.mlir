module {
  func.func public @fused_op_bmm_7214ee4e58801d7ac85c7b1d0ab51cefd28f2942_85x3x3xfloat64_85x3x1152xfloat64(%arg0: !torch.vtensor<[85,3,3],f64>, %arg1: !torch.vtensor<[85,3,1152],f64>) -> !torch.vtensor<[85,3,1152],f64> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[85,3,3],f64>, !torch.vtensor<[85,3,1152],f64> -> !torch.vtensor<[85,3,1152],f64>
    return %0 : !torch.vtensor<[85,3,1152],f64>
  }
}
