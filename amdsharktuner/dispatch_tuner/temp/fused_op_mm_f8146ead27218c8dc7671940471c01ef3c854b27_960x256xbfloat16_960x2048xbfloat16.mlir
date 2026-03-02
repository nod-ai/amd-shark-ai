module {
  func.func public @fused_op_mm_f8146ead27218c8dc7671940471c01ef3c854b27_960x256xbfloat16_960x2048xbfloat16(%arg0: !torch.vtensor<[960,256],bf16>, %arg1: !torch.vtensor<[960,2048],bf16>) -> !torch.vtensor<[256,2048],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[960,256],bf16>, !torch.list<int> -> !torch.vtensor<[256,960],bf16>
    %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[256,960],bf16>, !torch.vtensor<[960,2048],bf16> -> !torch.vtensor<[256,2048],bf16>
    return %2 : !torch.vtensor<[256,2048],bf16>
  }
}
