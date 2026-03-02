module {
  func.func public @fused_op_mm_51d6bbda6bbcffd4ca5d43b897baf26c1e3b853b_528x2304xbfloat16_2304x576xbfloat16(%arg0: !torch.vtensor<[528,2304],bf16>, %arg1: !torch.vtensor<[2304,576],bf16>) -> !torch.vtensor<[528,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[528,2304],bf16>, !torch.vtensor<[2304,576],bf16> -> !torch.vtensor<[528,576],bf16>
    return %0 : !torch.vtensor<[528,576],bf16>
  }
}
