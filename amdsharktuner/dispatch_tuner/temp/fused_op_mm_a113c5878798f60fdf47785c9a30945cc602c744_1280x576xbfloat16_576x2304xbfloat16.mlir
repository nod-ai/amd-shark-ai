module {
  func.func public @fused_op_mm_a113c5878798f60fdf47785c9a30945cc602c744_1280x576xbfloat16_576x2304xbfloat16(%arg0: !torch.vtensor<[1280,576],bf16>, %arg1: !torch.vtensor<[576,2304],bf16>) -> !torch.vtensor<[1280,2304],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1280,576],bf16>, !torch.vtensor<[576,2304],bf16> -> !torch.vtensor<[1280,2304],bf16>
    return %0 : !torch.vtensor<[1280,2304],bf16>
  }
}
