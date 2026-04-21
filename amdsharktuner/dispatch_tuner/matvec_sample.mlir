#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%arg0: tensor<4096x4096xf16>, %arg1: tensor<4096xf16>) -> tensor<4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %2 = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "reduction"]
    } ins(%arg0, %arg1 : tensor<4096x4096xf16>, tensor<4096xf16>)
      outs(%1 : tensor<4096xf32>) {
      ^bb0(%a: f16, %b: f16, %c: f32):
        %af = arith.extf %a : f16 to f32
        %bf = arith.extf %b : f16 to f32
        %m = arith.mulf %af, %bf : f32
        %acc = arith.addf %c, %m : f32
        linalg.yield %acc : f32
    } -> tensor<4096xf32>
    return %2 : tensor<4096xf32>
  }
}
