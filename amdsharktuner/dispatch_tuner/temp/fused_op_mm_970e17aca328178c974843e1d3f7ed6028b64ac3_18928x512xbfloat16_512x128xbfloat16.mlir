module {
  func.func public @fused_op_mm_970e17aca328178c974843e1d3f7ed6028b64ac3_18928x512xbfloat16_512x128xbfloat16(%arg0: !torch.vtensor<[18928,512],bf16>, %arg1: !torch.vtensor<[512,128],bf16>) -> !torch.vtensor<[18928,128],bf16> {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[18928,512],bf16>, !torch.vtensor<[512,128],bf16> -> !torch.vtensor<[18928,128],bf16>
    return %0 : !torch.vtensor<[18928,128],bf16>
  }
}
