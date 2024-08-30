// RUN: stablehlo-translate --interpret -split-input-file %s

module attributes {jax.uses_shape_polymorphism = true} {
  func.func @main() -> tensor<i1> {
    %cst = stablehlo.constant dense<[[[0.578237355], [-0.970109581], [-1.4660455], [-2.14128733], [-6.32983398], [3.11281967], [0.251084536], [-1.27283132], [-4.13759708], [2.18946671], [0.353660911], [-2.24349213], [-2.09625554], [2.21037674], [2.92937016], [7.15460968], [2.08107162], [0.570409536], [2.320630e+00], [1.12770963], [-0.655245662], [-0.193986982], [3.22981191], [-2.16140771], [0.855830848], [2.85118198], [-3.07397199], [1.75441694]]]> : tensor<1x28x1xf32>
    %cst_0 = stablehlo.constant dense<[[[3.96059895, -0.153480157, 3.08699727, -4.10661936, -4.03720093, 2.35196185, -0.688390791, 0.960256397, 0.248520851, -3.88998699, -0.347455829, -0.352612913, -1.53400397, -1.73838902, 3.71750188, 1.69945824]], [[-0.827514827, 2.37026954, -1.93849611, -3.15325975, 3.96820211, -4.9135704, -2.73580456, -2.29271531, -0.932672917, 0.120558962, 3.40987921, 0.155341223, -0.659520626, 0.0879687816, -0.620618999, -2.21843314]], [[7.13872862, 2.12502098, 2.49370933, 1.45718396, 2.15042424, -1.11660945, 0.214214519, 2.22703552, 1.08319771, -4.40505075, 5.33248425, -3.68989253, -1.15562749, -0.284276783, 3.43231511, -1.942080e-01]]]> : tensor<3x1x16xf32>
    %cst_1 = stablehlo.constant dense<"0x00000000A317153F0000000000000000A317153F00000000000000000000000000000000087A923DA317153FC053BC3D000000009D40513D000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000A317153F00000000A317153F0000000000000000A317153F000000006CDC0F3F087A123E0000000000000000000000000000000000000000A317153FA317153FC72A803FC72A803FC72A803FC72A803FC72A803F000000000BB75B3EC72A803FC72A803F00000000C72A803F000000000000000000000000C72A803F0000000062C8823E62C8823E62C8823E62C8823E62C8823E000000009D40513D62C8823E62C8823E0000000062C8823E00000000000000000000000062C8823E0000000000000000C72A803F0000000000000000C72A803F00000000000000000000000000000000561AFB3DC72A803F76F01C3E00000000E466A73D00000000000000000000000062C8823E000000000000000062C8823E00000000000000000000000000000000561AFB3C62C8823EE466273D00000000E466A73C0000000000000000C72A0040C72A803FC72A0040C72A803FC72A803FC72A803F0BB75B3E561AFB3F118E9F3F00000000C72A803F000000000000000000000000C72A0040C72A803FDA521A3F8918B73EDA521A3F8918B73E8918B73E62C8823E087A923D3FB5173FD47BD63E000000008918B73E000000000000000000000000DA521A3F62C8823E00000000C72A803F0000000000000000C72A803F00000000000000000000000000000000561AFB3DC72A803F76F01C3E00000000E466A73D0000000000000000000000008918B73E00000000000000008918B73E00000000000000000000000000000000E466273D8918B73E9D40513D00000000561AFB3C0000000000000000C72A0040C72A803FC72A0040C72A803FC72A803FC72A803F0BB75B3E561AFB3F118E9F3F00000000C72A803F000000000000000000000000C72A0040C72A803F1BA2AC3FC72A803F1BA2AC3FC72A803FC72A803F8918B73E0BB75B3E4D53AB3F35A18A3F00000000C72A803F0000000000000000000000001BA2AC3F8918B73EC72A803FC72A0040C72A803FC72A803FC72A0040000000000BB75B3EC72A803FC72A803F561AFB3DC72A004076F01C3E00000000E466A73DC72A803F00000000C72A803FC72A0040C72A803FC72A803FC72A0040000000000BB75B3EC72A803FC72A803F561AFB3DC72A004076F01C3E00000000E466A73DC72A803F00000000CA67C93FCA67C93FCA67C93F087A123FCA67C93FC72A803F561AFB3D932CC43F9D40513F561AFB3DCA67C93F76F01C3E00000000E466A73DCA67C93FC72A803FC72A0040C72A0040C72A0040C72A803FC72A0040C72A803F0BB75B3E561AFB3F118E9F3F561AFB3DC72A004076F01C3E00000000E466A73DC72A0040C72A803FC72A0040CA67C93FC72A0040C72A803FCA67C93FC72A803F0BB75B3E561AFB3F118E9F3F087A923DCA67C93FC053BC3D000000009D40513DC72A0040C72A803FC72A803FC72A803FC72A803F00000000C72A803FC72A803F000000001FDF753F561A7B3E561AFB3DC72A803F76F01C3E00000000E466A73DC72A803FC72A803F087A123FC72A803F087A123F00000000C72A803F087A123F00000000D03E0D3F087A123E561AFB3DC72A803F76F01C3E00000000E466A73D087A123F087A123FC72A0040C72A803FC72A0040C72A803FC72A803FC72A803F0BB75B3E561AFB3F118E9F3F00000000C72A803F000000000000000000000000C72A0040C72A803FC72A803F00000000C72A803F0000000000000000C72A803F000000001FDF753F561A7B3E0000000000000000000000000000000000000000C72A803FC72A803F0BB75B3F7EB7EC3F0BB75B3F0BB75B3F7EB7EC3F00000000C0533C3E0BB75B3F0BB75B3F561AFB3D7EB7EC3F76F01C3E00000000E466A73D0BB75B3F00000000C72A803FC72A803FC72A803FC72A803FC72A803F000000000BB75B3EC72A803FC72A803F00000000C72A803F000000000000000000000000C72A803F00000000C72A803F0BB75B3FC72A803F000000000BB75B3FC72A803F000000001FDF753F561A7B3E9D40D13D0BB75B3F9903083E00000000087A923DC72A803FC72A803FC72A803FC72A0040C72A803FC72A803FC72A0040000000000BB75B3EC72A803FC72A803F561AFB3DC72A004076F01C3E00000000E466A73DC72A803F000000000BB75B3F000000000BB75B3F00000000000000000BB75B3F000000009D40513F0BB75B3E00000000000000000000000000000000000000000BB75B3F0BB75B3FC72A803FC72A803FC72A803F00000000C72A803FC72A803F000000001FDF753F561A7B3E561AFB3DC72A803F76F01C3E00000000E466A73DC72A803FC72A803F"> : tensor<1x28x16xf32>
    %0 = stablehlo.uniform_quantize %cst_0 : (tensor<3x1x16xf32>) -> tensor<3x1x16x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>
    %1 = stablehlo.uniform_quantize %cst : (tensor<1x28x1xf32>) -> tensor<1x28x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>
    %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {pad = [[2, 2]], rhs_dilate = [2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x28x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>, tensor<3x1x16x!quant.uniform<i8:f32, 0.0039215482917486456:-128>>) -> tensor<1x28x16x!quant.uniform<i32:f32, 1.5367804432676217E-5>>
    %3 = stablehlo.uniform_quantize %2 : (tensor<1x28x16x!quant.uniform<i32:f32, 1.5367804432676217E-5>>) -> tensor<1x28x16x!quant.uniform<i8:f32, 0.0102174020281025:-128>>
    %4 = stablehlo.uniform_dequantize %3 : (tensor<1x28x16x!quant.uniform<i8:f32, 0.0102174020281025:-128>>) -> tensor<1x28x16xf32>
    %5 = stablehlo.custom_call @check.eq(%cst_1, %4) : (tensor<1x28x16xf32>, tensor<1x28x16xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
}