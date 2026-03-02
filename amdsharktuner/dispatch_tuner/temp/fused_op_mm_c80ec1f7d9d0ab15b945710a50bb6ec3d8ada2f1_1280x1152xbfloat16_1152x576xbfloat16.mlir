module {
  func.func public @fused_op_mm_c80ec1f7d9d0ab15b945710a50bb6ec3d8ada2f1_1280x1152xbfloat16_1152x576xbfloat16(%arg0: !torch.vtensor<[1280,1152],bf16>, %arg1: !torch.vtensor<[1152,576],bf16>) -> !torch.vtensor<[1280,576],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1280,1152],bf16>, !torch.vtensor<[1152,576],bf16> -> !torch.vtensor<[1280,576],bf16>
    return %0 : !torch.vtensor<[1280,576],bf16>
  }
}
