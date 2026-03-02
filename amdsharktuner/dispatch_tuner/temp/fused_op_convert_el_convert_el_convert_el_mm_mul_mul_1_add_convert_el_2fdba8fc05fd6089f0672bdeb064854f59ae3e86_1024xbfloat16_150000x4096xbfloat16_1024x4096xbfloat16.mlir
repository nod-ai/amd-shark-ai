module {
  func.func public @fused_op_convert_el_convert_el_convert_el_mm_mul_mul_1_add_convert_el_2fdba8fc05fd6089f0672bdeb064854f59ae3e86_1024xbfloat16_150000x4096xbfloat16_1024x4096xbfloat16(%arg0: !torch.vtensor<[1024],bf16>, %arg1: !torch.vtensor<[150000,4096],bf16>, %arg2: !torch.vtensor<[1024,4096],bf16>) -> !torch.vtensor<[150000,1024],bf16> {
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg2, %0 : !torch.vtensor<[1024,4096],bf16>, !torch.list<int> -> !torch.vtensor<[4096,1024],bf16>
    %int6 = torch.constant.int 6
    %2 = torch.prims.convert_element_type %arg0, %int6 : !torch.vtensor<[1024],bf16>, !torch.int -> !torch.vtensor<[1024],f32>
    %int6_0 = torch.constant.int 6
    %3 = torch.prims.convert_element_type %arg1, %int6_0 : !torch.vtensor<[150000,4096],bf16>, !torch.int -> !torch.vtensor<[150000,4096],f32>
    %int6_1 = torch.constant.int 6
    %4 = torch.prims.convert_element_type %1, %int6_1 : !torch.vtensor<[4096,1024],bf16>, !torch.int -> !torch.vtensor<[4096,1024],f32>
    %5 = torch.aten.mm %3, %4 : !torch.vtensor<[150000,4096],f32>, !torch.vtensor<[4096,1024],f32> -> !torch.vtensor<[150000,1024],f32>
    %int1_2 = torch.constant.int 1
    %6 = torch.aten.mul.Scalar %5, %int1_2 : !torch.vtensor<[150000,1024],f32>, !torch.int -> !torch.vtensor<[150000,1024],f32>
    %int1_3 = torch.constant.int 1
    %7 = torch.aten.mul.Scalar %2, %int1_3 : !torch.vtensor<[1024],f32>, !torch.int -> !torch.vtensor<[1024],f32>
    %int1_4 = torch.constant.int 1
    %8 = torch.aten.add.Tensor %6, %7, %int1_4 : !torch.vtensor<[150000,1024],f32>, !torch.vtensor<[1024],f32>, !torch.int -> !torch.vtensor<[150000,1024],f32>
    %int15 = torch.constant.int 15
    %9 = torch.prims.convert_element_type %8, %int15 : !torch.vtensor<[150000,1024],f32>, !torch.int -> !torch.vtensor<[150000,1024],bf16>
    return %9 : !torch.vtensor<[150000,1024],bf16>
  }
}
