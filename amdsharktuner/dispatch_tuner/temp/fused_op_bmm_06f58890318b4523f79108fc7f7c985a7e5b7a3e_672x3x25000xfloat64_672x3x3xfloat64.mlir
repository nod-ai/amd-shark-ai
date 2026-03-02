module {
  func.func public @fused_op_bmm_06f58890318b4523f79108fc7f7c985a7e5b7a3e_672x3x25000xfloat64_672x3x3xfloat64(%arg0: !torch.vtensor<[672,3,25000],f64>, %arg1: !torch.vtensor<[672,3,3],f64>) -> !torch.vtensor<[672,25000,3],f64> {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int2, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[672,3,25000],f64>, !torch.list<int> -> !torch.vtensor<[672,25000,3],f64>
    %int0_0 = torch.constant.int 0
    %int2_1 = torch.constant.int 2
    %int1_2 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int0_0, %int2_1, %int1_2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.permute %arg1, %2 : !torch.vtensor<[672,3,3],f64>, !torch.list<int> -> !torch.vtensor<[672,3,3],f64>
    %4 = torch.aten.bmm %1, %3 : !torch.vtensor<[672,25000,3],f64>, !torch.vtensor<[672,3,3],f64> -> !torch.vtensor<[672,25000,3],f64>
    return %4 : !torch.vtensor<[672,25000,3],f64>
  }
}
