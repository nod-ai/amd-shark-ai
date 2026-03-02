module {
  func.func public @fused_op_mm_baddcb06bbf44da835a11de349cfc1a2e6030a64_32768x512xbfloat16_32768x128xbfloat16(%arg0: !torch.vtensor<[32768,512],bf16>, %arg1: !torch.vtensor<[32768,128],bf16>) -> !torch.vtensor<[512,128],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[32768,512],bf16>, !torch.list<int> -> !torch.vtensor<[512,32768],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[512,32768],bf16>, !torch.vtensor<[32768,128],bf16> -> !torch.vtensor<[512,128],bf16>
    return %2 : !torch.vtensor<[512,128],bf16>
  }
}
