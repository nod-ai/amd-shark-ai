module {
  func.func public @fused_op_mm_b563e01b779abe2125c1011159012e603a8d12a5_1285x2048xbfloat16_3840x2048xbfloat16(%arg0: !torch.vtensor<[1285,2048],bf16>, %arg1: !torch.vtensor<[3840,2048],bf16>) -> !torch.vtensor<[1285,3840],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[3840,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,3840],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[1285,2048],bf16>, !torch.vtensor<[2048,3840],bf16> -> !torch.vtensor<[1285,3840],bf16>
    return %2 : !torch.vtensor<[1285,3840],bf16>
  }
}
