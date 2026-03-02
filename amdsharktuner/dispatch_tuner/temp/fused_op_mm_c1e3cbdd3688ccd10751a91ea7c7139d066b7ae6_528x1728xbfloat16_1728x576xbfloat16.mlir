module {
  func.func public @fused_op_mm_c1e3cbdd3688ccd10751a91ea7c7139d066b7ae6_528x1728xbfloat16_1728x576xbfloat16(%arg0: !torch.vtensor<[528,1728],bf16>, %arg1: !torch.vtensor<[1728,576],bf16>) -> !torch.vtensor<[528,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[528,1728],bf16>, !torch.vtensor<[1728,576],bf16> -> !torch.vtensor<[528,576],bf16>
    return %0 : !torch.vtensor<[528,576],bf16>
  }
}
