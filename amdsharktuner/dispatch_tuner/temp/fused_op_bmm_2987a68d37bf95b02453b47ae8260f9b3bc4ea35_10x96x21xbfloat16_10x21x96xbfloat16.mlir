module {
  func.func public @fused_op_bmm_2987a68d37bf95b02453b47ae8260f9b3bc4ea35_10x96x21xbfloat16_10x21x96xbfloat16(%arg0: !torch.vtensor<[10,96,21],bf16>, %arg1: !torch.vtensor<[10,21,96],bf16>) -> !torch.vtensor<[10,96,96],bf16> {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[10,96,21],bf16>, !torch.vtensor<[10,21,96],bf16> -> !torch.vtensor<[10,96,96],bf16>
    return %0 : !torch.vtensor<[10,96,96],bf16>
  }
}
