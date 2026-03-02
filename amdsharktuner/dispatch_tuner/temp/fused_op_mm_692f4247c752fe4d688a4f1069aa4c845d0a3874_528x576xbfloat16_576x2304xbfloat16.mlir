module {
  func.func public @fused_op_mm_692f4247c752fe4d688a4f1069aa4c845d0a3874_528x576xbfloat16_576x2304xbfloat16(%arg0: !torch.vtensor<[528,576],bf16>, %arg1: !torch.vtensor<[576,2304],bf16>) -> !torch.vtensor<[528,2304],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[528,576],bf16>, !torch.vtensor<[576,2304],bf16> -> !torch.vtensor<[528,2304],bf16>
    return %0 : !torch.vtensor<[528,2304],bf16>
  }
}
