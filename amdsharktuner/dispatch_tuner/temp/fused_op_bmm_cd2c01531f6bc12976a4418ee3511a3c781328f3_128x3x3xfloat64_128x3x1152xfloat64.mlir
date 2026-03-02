module {
  func.func public @fused_op_bmm_cd2c01531f6bc12976a4418ee3511a3c781328f3_128x3x3xfloat64_128x3x1152xfloat64(%arg0: !torch.vtensor<[128,3,3],f64>, %arg1: !torch.vtensor<[128,3,1152],f64>) -> !torch.vtensor<[128,3,1152],f64> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[128,3,3],f64>, !torch.vtensor<[128,3,1152],f64> -> !torch.vtensor<[128,3,1152],f64>
    return %0 : !torch.vtensor<[128,3,1152],f64>
  }
}
