// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<12x12xf32>
    %1 = call @expected() : () -> tensor<6x6xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, window_dimensions = dense<3> : tensor<2xi64>, window_strides = dense<2> : tensor<2xi64>} : (tensor<12x12xf32>, tensor<f32>) -> tensor<6x6xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<6x6xf32>, tensor<6x6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<12x12xf32> {
    %0 = stablehlo.constant dense<"0x95ED25BF92CCE1BF717A9E3F03D954C005BEA94030B60040C0BC604091C9A6BFDE0742C023955CBDE536C23F613F01C01AFF9FBFDB99BD3FE3235CC0B3F804C09B4B5EC0A658094095EF65C0C9002940E41C11C0912D56C0C035144016A2D83F559B5E3FF7238940EE9EB5BD3C6FE8BFAC52A1C0FE62C2C0B739213F2E35E33F98F341C068FE854000F2534076308D404681D6BF9EBAD2BF6D1F9DC0F4DFC3BFD755793DCDEB06406536503FE7B8EF3E784438BF564F46407C7D5EBFB5B0194006F080408F2385BEAC7D84400F7437C04EDA9EBF424F25409DB7143F46428EBFC9591E3FA585EEBF1A720BBFAE570CBF9191E940CEFAAF400926BBBF443158C03B7ACA3E462384C0C3C6C53FFE4FDA3D89CA57C02E7E75BEF55A29BFE41A2D3F45142040390B32404F157DBDE387CEBF159CE3BF5D80CDC0D7A314400F8160C003279CBE6320B140819E32C03855E43F621459BF35B8DFBE1D372B408E1D06C0D5D486BFB9D4D73E361F8540FE801A4003B3B23E9031E03F5239C93F3AAC06BF92887FBFDD0DAD3F7A723EBF89CDCFC01FD91740A49C3DBF20F75BBE27110F3F47A90C40CFDA44C0E705A740FC0A833F6CF03D3E8EB47EC09F43A63F20796CC06C6D2EC0EFC1C4C0B17B51C03A7B1340B1991C3E147E44BF01865A3E6CD0F9BF647557BD1E5153C0A0F6F340893530BFCACE0040836883BE97454F3F66E7F7BFF78D063F23FB3140CE9A634033572840AF171B40CDDAB0C067F7A9C0EE995D3F3FBBF83FA54E703D08E9BA3F468F87409AC5A4403DBE7C40B6CED93ED01A9140"> : tensor<12x12xf32>
    return %0 : tensor<12x12xf32>
  }
  func.func private @expected() -> tensor<6x6xf32> {
    %0 = stablehlo.constant dense<[[0.684010506, -12.719841, -4.57913065, -4.66834545, -0.397963524, 11.229845], [4.74389315, -13.2925587, -5.58094263, 0.024184227, 4.18854761, 8.16319561], [24.4672756, -7.86530733, -6.14348221, -3.12228608, -3.62333155, -2.08534884], [6.22157717, -8.79234218, -0.932239771, 7.97472048, 10.4533367, 6.282166], [1.38395691, -1.03837013, -8.16451168, 1.12453938, 10.8031111, 10.691618], [-4.14541149, 6.4455533, 6.027740e+00, 10.2441072, 16.3865185, 11.146574]]> : tensor<6x6xf32>
    return %0 : tensor<6x6xf32>
  }
}
