// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4xf32>, tensor<4xf32>)
    %1 = call @expected() : () -> tensor<f32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.constant dense<[-1.73818827, 6.32115507, 2.81545162, -1.37914991]> : tensor<4xf32>
    %1 = stablehlo.constant dense<[-4.02553225, -2.70646834, 3.14252234, 1.59961236]> : tensor<4xf32>
    return %0, %1 : tensor<4xf32>, tensor<4xf32>
  }
  func.func private @expected() -> tensor<f32> {
    %0 = stablehlo.constant dense<-3.46935892> : tensor<f32>
    return %0 : tensor<f32>
  }
}
