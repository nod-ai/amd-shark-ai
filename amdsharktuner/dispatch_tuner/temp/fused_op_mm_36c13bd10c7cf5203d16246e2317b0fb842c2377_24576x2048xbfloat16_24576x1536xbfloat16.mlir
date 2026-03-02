module {
  func.func public @fused_op_mm_36c13bd10c7cf5203d16246e2317b0fb842c2377_24576x2048xbfloat16_24576x1536xbfloat16(%arg0: !torch.vtensor<[24576,2048],bf16>, %arg1: !torch.vtensor<[24576,1536],bf16>) -> !torch.vtensor<[2048,1536],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[24576,2048],bf16>, !torch.list<int> -> !torch.vtensor<[2048,24576],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2048,24576],bf16>, !torch.vtensor<[24576,1536],bf16> -> !torch.vtensor<[2048,1536],bf16>
    return %2 : !torch.vtensor<[2048,1536],bf16>
  }
}
