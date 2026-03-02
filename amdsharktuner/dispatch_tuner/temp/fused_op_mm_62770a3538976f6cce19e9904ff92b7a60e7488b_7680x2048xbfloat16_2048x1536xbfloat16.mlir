module {
  func.func public @fused_op_mm_62770a3538976f6cce19e9904ff92b7a60e7488b_7680x2048xbfloat16_2048x1536xbfloat16(%arg0: !torch.vtensor<[7680,2048],bf16>, %arg1: !torch.vtensor<[2048,1536],bf16>) -> !torch.vtensor<[7680,1536],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[7680,2048],bf16>, !torch.vtensor<[2048,1536],bf16> -> !torch.vtensor<[7680,1536],bf16>
    return %0 : !torch.vtensor<[7680,1536],bf16>
  }
}
