// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xi64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xi64>, tensor<5x2x2xi64>)
    %1 = call @expected() : () -> tensor<5x6x7xi64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<i64>
      stablehlo.return %3 : tensor<i64>
    }) : (tensor<5x6x7xi64>, tensor<2x2x2xi64>, tensor<5x2x2xi64>) -> tensor<5x6x7xi64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi64>, tensor<5x6x7xi64>) -> ()
    return %2 : tensor<5x6x7xi64>
  }
  func.func private @inputs() -> (tensor<5x6x7xi64> {mhlo.layout_mode = "default"}, tensor<5x2x2xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0100000000000000FFFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0000000000000000FBFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0200000000000000000000000000000001000000000000000000000000000000000000000000000001000000000000000000000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000040000000000000000000000000000000300000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF010000000000000001000000000000000100000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF050000000000000004000000000000000100000000000000FDFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000200000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FDFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0100000000000000020000000000000000000000000000000200000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000300000000000000FCFFFFFFFFFFFFFF000000000000000000000000000000000300000000000000FDFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0600000000000000FCFFFFFFFFFFFFFF040000000000000002000000000000000500000000000000FEFFFFFFFFFFFFFF0600000000000000FEFFFFFFFFFFFFFF00000000000000000000000000000000FEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FBFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF00000000000000000000000000000000020000000000000003000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFF0000000000000000FAFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFF04000000000000000400000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0300000000000000FAFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000000002000000000000000000000000000000FDFFFFFFFFFFFFFF02000000000000000200000000000000FFFFFFFFFFFFFFFF040000000000000003000000000000000100000000000000060000000000000003000000000000000200000000000000020000000000000000000000000000000000000000000000FCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0000000000000000040000000000000001000000000000000000000000000000FFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0100000000000000FCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF01000000000000000300000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000800000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000FEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF050000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FAFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0400000000000000FCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0500000000000000FFFFFFFFFFFFFFFF01000000000000000300000000000000FEFFFFFFFFFFFFFF00000000000000000100000000000000FEFFFFFFFFFFFFFFFAFFFFFFFFFFFFFF06000000000000000000000000000000FBFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFF0300000000000000010000000000000004000000000000000000000000000000FFFFFFFFFFFFFFFF010000000000000005000000000000000100000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF"> : tensor<5x6x7xi64>
    %c_0 = stablehlo.constant dense<[[[-4, 0], [-2, -2]], [[0, -2], [4, 6]], [[2, 0], [3, -2]], [[0, 0], [0, -1]], [[7, 2], [5, 0]]]> : tensor<5x2x2xi64>
    return %c, %c_0 : tensor<5x6x7xi64>, tensor<5x2x2xi64>
  }
  func.func private @expected() -> (tensor<5x6x7xi64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0100000000000000FBFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0000000000000000FBFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFF01000000000000000000000000000000000000000000000001000000000000000000000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000040000000000000000000000000000000300000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF01000000000000000100000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF050000000000000004000000000000000100000000000000FDFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000200000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FDFFFFFFFFFFFFFF000000000000000004000000000000000100000000000000020000000000000000000000000000000200000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0300000000000000FCFFFFFFFFFFFFFF000000000000000000000000000000000300000000000000FDFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0A00000000000000FCFFFFFFFFFFFFFF040000000000000002000000000000000500000000000000FEFFFFFFFFFFFFFF0600000000000000FEFFFFFFFFFFFFFF00000000000000000000000000000000FEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FBFFFFFFFFFFFFFF0200000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF00000000000000000000000000000000000000000000000003000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFF0000000000000000FAFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFF04000000000000000400000000000000FFFFFFFFFFFFFFFF00000000000000000300000000000000FAFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000000002000000000000000000000000000000FDFFFFFFFFFFFFFF02000000000000000200000000000000FFFFFFFFFFFFFFFF040000000000000003000000000000000100000000000000060000000000000003000000000000000200000000000000020000000000000000000000000000000000000000000000FCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0000000000000000040000000000000001000000000000000000000000000000FFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0100000000000000FCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF01000000000000000300000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000800000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000FEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF050000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF04000000000000000000000000000000FAFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0400000000000000FCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0500000000000000FFFFFFFFFFFFFFFF01000000000000000300000000000000000000000000000000000000000000000100000000000000FEFFFFFFFFFFFFFFFAFFFFFFFFFFFFFF06000000000000000000000000000000FBFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000300000000000000FEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFF0300000000000000010000000000000004000000000000000000000000000000FFFFFFFFFFFFFFFF010000000000000005000000000000000100000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF"> : tensor<5x6x7xi64>
    return %c : tensor<5x6x7xi64>
  }
}