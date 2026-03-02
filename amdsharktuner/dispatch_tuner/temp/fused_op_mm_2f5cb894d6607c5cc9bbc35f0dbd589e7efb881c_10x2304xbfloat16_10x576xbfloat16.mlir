module {
  func.func public @fused_op_mm_2f5cb894d6607c5cc9bbc35f0dbd589e7efb881c_10x2304xbfloat16_10x576xbfloat16(%arg0: !torch.vtensor<[10,2304],bf16>, %arg1: !torch.vtensor<[10,576],bf16>) -> !torch.vtensor<[2304,576],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[10,2304],bf16>, !torch.list<int> -> !torch.vtensor<[2304,10],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2304,10],bf16>, !torch.vtensor<[10,576],bf16> -> !torch.vtensor<[2304,576],bf16>
    return %2 : !torch.vtensor<[2304,576],bf16>
  }
}
