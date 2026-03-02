module {
  func.func public @fused_op_mm_4dc8353f99c825d0383875865760b7d858dba00d_7680x576xbfloat16_1728x576xbfloat16(%arg0: !torch.vtensor<[7680,576],bf16>, %arg1: !torch.vtensor<[1728,576],bf16>) -> !torch.vtensor<[7680,1728],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[1728,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,1728],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[7680,576],bf16>, !torch.vtensor<[576,1728],bf16> -> !torch.vtensor<[7680,1728],bf16>
    return %2 : !torch.vtensor<[7680,1728],bf16>
  }
}
