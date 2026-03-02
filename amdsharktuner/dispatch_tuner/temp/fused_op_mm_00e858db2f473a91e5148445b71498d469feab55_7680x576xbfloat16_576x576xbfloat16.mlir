module {
  func.func public @fused_op_mm_00e858db2f473a91e5148445b71498d469feab55_7680x576xbfloat16_576x576xbfloat16(%arg0: !torch.vtensor<[7680,576],bf16>, %arg1: !torch.vtensor<[576,576],bf16>) -> !torch.vtensor<[7680,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[7680,576],bf16>, !torch.vtensor<[576,576],bf16> -> !torch.vtensor<[7680,576],bf16>
    return %0 : !torch.vtensor<[7680,576],bf16>
  }
}
