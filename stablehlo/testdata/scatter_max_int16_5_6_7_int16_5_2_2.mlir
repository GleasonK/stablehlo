// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xi16>, tensor<5x2x2xi16>)
    %1 = call @expected() : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<5x6x7xi16>, tensor<2x2x2xi64>, tensor<5x2x2xi16>) -> tensor<5x6x7xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    return %2 : tensor<5x6x7xi16>
  }
  func.func private @inputs() -> (tensor<5x6x7xi16> {mhlo.layout_mode = "default"}, tensor<5x2x2xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00000000FDFF01000000FCFFF9FF0100FDFFFEFFFEFFFEFF010000000000FFFFFCFF04000100000000000100FDFF00000400010002000100020002000600000000000300FEFF000004000800000001000000FFFF0100FFFFFCFF0100FFFFFEFF03000600030003000000FCFF0300FCFF020002000000FFFF00000000000001000000FDFFFEFF01000300FFFF00000300FDFF03000400010000000400FEFFFFFF010001000200020004000100FFFF0000020001000200FFFF0000FFFF03000400FDFFFEFF01000200FEFF010003000000040001000200020000000000FFFFFEFF0200000003000000FFFF03000400000001000200010004000100000001000100FCFF0100060000000100FCFFFEFF0200FFFFFCFFFBFFFDFF000003000100FFFFFDFF05000100FFFF050000000200020002000000000000000300FFFFFBFF0000FCFF0200FAFF0400FFFFFEFFFBFF0000FEFFFDFFFFFFFFFFFEFF00000000000000000400FFFFFEFFFBFF0200000001000500FDFFFDFFFFFFFBFFFFFFFFFFFFFF00000000FFFF00000300FAFF0000FCFF00000000FCFFFFFFFEFFFFFF0000FBFFFEFFFDFF"> : tensor<5x6x7xi16>
    %c_0 = stablehlo.constant dense<[[[2, 0], [0, -2]], [[0, 12], [-1, 0]], [[0, -4], [-2, -5]], [[2, -1], [5, -1]], [[0, -5], [4, -2]]]> : tensor<5x2x2xi16>
    return %c, %c_0 : tensor<5x6x7xi16>, tensor<5x2x2xi16>
  }
  func.func private @expected() -> (tensor<5x6x7xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00000200FDFF01000000FCFFF9FF0100FDFFFEFFFEFFFEFF010000000000FFFFFCFF04000100000000000100FDFF00000400010002000100020002000600000000000300FEFF000004000800000001000000FFFF01000000FCFF0100FFFFFEFF03000600030003000000FCFF0300FCFF0200020000000C0000000000000001000000FDFFFEFF01000300FFFF00000300FDFF03000400010000000400FEFFFFFF010001000200020004000100FFFF0000020001000200FFFF0000FFFF03000400FDFFFEFF01000200FEFF010003000000040001000200020000000000FFFFFEFF0200000003000000FFFF03000400000001000200010004000100000001000200FCFF0100060000000100FCFFFEFF0200FFFFFCFFFBFFFDFF000003000100FFFFFDFF05000100FFFF050000000200020002000000050000000300FFFFFBFF0000FCFF0200FAFF0400FFFFFEFFFBFF0000FEFF0000FFFFFFFFFEFF00000000000000000400FFFFFEFFFBFF0200000001000500FDFFFDFFFFFFFBFFFFFFFFFFFFFF00000000FFFF00000400FAFF0000FCFF00000000FCFFFFFFFEFFFFFF0000FBFFFEFFFDFF"> : tensor<5x6x7xi16>
    return %c : tensor<5x6x7xi16>
  }
}
