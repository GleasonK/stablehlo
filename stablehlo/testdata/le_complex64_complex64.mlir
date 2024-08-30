// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i1> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<complex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<i1>
    %2 = stablehlo.real %0#0 : (tensor<complex<f32>>) -> tensor<f32>
    %3 = stablehlo.real %0#1 : (tensor<complex<f32>>) -> tensor<f32>
    %4 = stablehlo.compare  EQ, %2, %3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.imag %0#0 : (tensor<complex<f32>>) -> tensor<f32>
    %6 = stablehlo.imag %0#1 : (tensor<complex<f32>>) -> tensor<f32>
    %7 = stablehlo.compare  LE, %5, %6,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = stablehlo.real %0#0 : (tensor<complex<f32>>) -> tensor<f32>
    %9 = stablehlo.real %0#1 : (tensor<complex<f32>>) -> tensor<f32>
    %10 = stablehlo.compare  LE, %8, %9,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %11 = stablehlo.select %4, %7, %10 : tensor<i1>, tensor<i1>
    stablehlo.custom_call @check.expect_eq(%11, %1) {has_side_effect = true} : (tensor<i1>, tensor<i1>) -> ()
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<complex<f32>> {mhlo.layout_mode = "default"}, tensor<complex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(1.10811841,-6.992440e-02)> : tensor<complex<f32>>
    %cst_0 = stablehlo.constant dense<(3.3754487,-2.81067419)> : tensor<complex<f32>>
    return %cst, %cst_0 : tensor<complex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> (tensor<i1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<i1>
    return %c : tensor<i1>
  }
}