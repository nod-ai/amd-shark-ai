module {
  func.func public @fused_op_mm_a3b0ea6a0f51b8e4a2860aff1e387a7521bb3fd2_528x1728xbfloat16_528x576xbfloat16(%arg0: !torch.vtensor<[528,1728],bf16>, %arg1: !torch.vtensor<[528,576],bf16>) -> !torch.vtensor<[1728,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[528,1728],bf16>, !torch.list<int> -> !torch.vtensor<[1728,528],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[1728,528],bf16>, !torch.vtensor<[528,576],bf16> -> !torch.vtensor<[1728,576],bf16>
    return %2 : !torch.vtensor<[1728,576],bf16>
  }
}
