module {
  func.func public @fused_op_mm_920ddc462c3e35485abd0b2864859dd40f992465_24576x1728xbfloat16_24576x576xbfloat16(%arg0: !torch.vtensor<[24576,1728],bf16>, %arg1: !torch.vtensor<[24576,576],bf16>) -> !torch.vtensor<[1728,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[24576,1728],bf16>, !torch.list<int> -> !torch.vtensor<[1728,24576],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[1728,24576],bf16>, !torch.vtensor<[24576,576],bf16> -> !torch.vtensor<[1728,576],bf16>
    return %2 : !torch.vtensor<[1728,576],bf16>
  }
}
