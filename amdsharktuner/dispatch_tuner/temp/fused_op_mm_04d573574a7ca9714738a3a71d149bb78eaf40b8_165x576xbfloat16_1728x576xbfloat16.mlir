module {
  func.func public @fused_op_mm_04d573574a7ca9714738a3a71d149bb78eaf40b8_165x576xbfloat16_1728x576xbfloat16(%arg0: !torch.vtensor<[165,576],bf16>, %arg1: !torch.vtensor<[1728,576],bf16>) -> !torch.vtensor<[165,1728],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg1, %0 : !torch.vtensor<[1728,576],bf16>, !torch.list<int> -> !torch.vtensor<[576,1728],bf16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[165,576],bf16>, !torch.vtensor<[576,1728],bf16> -> !torch.vtensor<[165,1728],bf16>
    return %2 : !torch.vtensor<[165,1728],bf16>
  }
}
