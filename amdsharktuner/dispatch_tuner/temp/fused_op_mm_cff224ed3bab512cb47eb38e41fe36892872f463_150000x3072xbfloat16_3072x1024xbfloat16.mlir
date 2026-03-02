module {
  func.func public @fused_op_mm_cff224ed3bab512cb47eb38e41fe36892872f463_150000x3072xbfloat16_3072x1024xbfloat16(%arg0: !torch.vtensor<[150000,3072],bf16>, %arg1: !torch.vtensor<[3072,1024],bf16>) -> !torch.vtensor<[150000,1024],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[150000,3072],bf16>, !torch.vtensor<[3072,1024],bf16> -> !torch.vtensor<[150000,1024],bf16>
    return %0 : !torch.vtensor<[150000,1024],bf16>
  }
}
