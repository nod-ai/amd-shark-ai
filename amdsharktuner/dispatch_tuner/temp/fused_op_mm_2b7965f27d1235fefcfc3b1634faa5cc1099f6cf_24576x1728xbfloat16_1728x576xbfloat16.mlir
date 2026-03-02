module {
  func.func public @fused_op_mm_2b7965f27d1235fefcfc3b1634faa5cc1099f6cf_24576x1728xbfloat16_1728x576xbfloat16(%arg0: !torch.vtensor<[24576,1728],bf16>, %arg1: !torch.vtensor<[1728,576],bf16>) -> !torch.vtensor<[24576,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[24576,1728],bf16>, !torch.vtensor<[1728,576],bf16> -> !torch.vtensor<[24576,576],bf16>
    return %0 : !torch.vtensor<[24576,576],bf16>
  }
}
