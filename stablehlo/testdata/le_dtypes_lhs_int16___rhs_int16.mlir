// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<i16>, tensor<i16>)
    %1 = call @expected() : () -> tensor<i1>
    %2 = stablehlo.compare  LE, %0#0, %0#1,  SIGNED : (tensor<i16>, tensor<i16>) -> tensor<i1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<i16>, tensor<i16>) {
    %0 = stablehlo.constant dense<-4> : tensor<i16>
    %1 = stablehlo.constant dense<2> : tensor<i16>
    return %0, %1 : tensor<i16>, tensor<i16>
  }
  func.func private @expected() -> tensor<i1> {
    %0 = stablehlo.constant dense<true> : tensor<i1>
    return %0 : tensor<i1>
  }
}