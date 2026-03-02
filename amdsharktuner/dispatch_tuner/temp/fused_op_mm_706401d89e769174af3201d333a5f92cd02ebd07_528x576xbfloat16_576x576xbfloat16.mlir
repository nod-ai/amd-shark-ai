module {
  func.func public @fused_op_mm_706401d89e769174af3201d333a5f92cd02ebd07_528x576xbfloat16_576x576xbfloat16(%arg0: !torch.vtensor<[528,576],bf16>, %arg1: !torch.vtensor<[576,576],bf16>) -> !torch.vtensor<[528,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[528,576],bf16>, !torch.vtensor<[576,576],bf16> -> !torch.vtensor<[528,576],bf16>
    return %0 : !torch.vtensor<[528,576],bf16>
  }
}
