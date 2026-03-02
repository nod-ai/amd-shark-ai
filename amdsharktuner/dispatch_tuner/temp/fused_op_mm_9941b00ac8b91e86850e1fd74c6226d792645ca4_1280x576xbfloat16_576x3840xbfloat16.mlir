module {
  func.func public @fused_op_mm_9941b00ac8b91e86850e1fd74c6226d792645ca4_1280x576xbfloat16_576x3840xbfloat16(%arg0: !torch.vtensor<[1280,576],bf16>, %arg1: !torch.vtensor<[576,3840],bf16>) -> !torch.vtensor<[1280,3840],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1280,576],bf16>, !torch.vtensor<[576,3840],bf16> -> !torch.vtensor<[1280,3840],bf16>
    return %0 : !torch.vtensor<[1280,3840],bf16>
  }
}
