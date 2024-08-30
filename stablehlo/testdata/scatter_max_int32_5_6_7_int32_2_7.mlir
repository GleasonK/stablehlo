// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xi32>, tensor<2x7xi32>)
    %1 = call @expected() : () -> tensor<5x6x7xi32>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i32>
      stablehlo.return %3 : tensor<i32>
    }) : (tensor<5x6x7xi32>, tensor<2x2xi64>, tensor<2x7xi32>) -> tensor<5x6x7xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xi32>, tensor<5x6x7xi32>) -> ()
    return %2 : tensor<5x6x7xi32>
  }
  func.func private @inputs() -> (tensor<5x6x7xi32> {mhlo.layout_mode = "default"}, tensor<2x7xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0200000002000000FCFFFFFF0100000000000000020000000000000001000000FDFFFFFFF9FFFFFF0000000000000000010000000100000001000000F8FFFFFF030000000200000000000000FFFFFFFF02000000FFFFFFFF04000000010000000000000002000000FEFFFFFFFDFFFFFFFEFFFFFFFEFFFFFF01000000FDFFFFFF00000000FDFFFFFF01000000000000000000000002000000010000000700000002000000010000000300000003000000FFFFFFFF00000000010000000000000003000000FEFFFFFFFDFFFFFF000000000100000002000000FDFFFFFF00000000FFFFFFFFFFFFFFFF0100000005000000FDFFFFFF01000000FDFFFFFFFFFFFFFF000000000300000002000000030000000000000003000000020000000100000007000000F7FFFFFF0300000003000000FEFFFFFF01000000FDFFFFFF0400000004000000FEFFFFFFFDFFFFFF02000000FDFFFFFF0000000001000000FFFFFFFFFFFFFFFF040000000000000000000000FFFFFFFF000000000000000005000000FEFFFFFF0100000001000000FDFFFFFF01000000FFFFFFFF01000000F8FFFFFF010000000000000000000000FFFFFFFF0100000000000000FEFFFFFF000000000000000001000000FEFFFFFF02000000FCFFFFFF00000000FEFFFFFF020000000300000000000000020000000000000005000000FFFFFFFF02000000FFFFFFFF03000000FEFFFFFF00000000FDFFFFFF040000000400000000000000000000000200000003000000FDFFFFFF020000000100000003000000FFFFFFFF01000000FDFFFFFFFDFFFFFF0600000000000000FEFFFFFF03000000FFFFFFFF04000000FDFFFFFFFEFFFFFFF8FFFFFF0300000001000000FAFFFFFF030000000000000000000000FFFFFFFFFCFFFFFF000000000000000000000000FFFFFFFF0000000001000000000000000400000000000000FBFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFAFFFFFFFCFFFFFFFCFFFFFF000000000000000004000000FEFFFFFF000000000200000000000000FDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF01000000FFFFFFFFFEFFFFFF00000000FFFFFFFF00000000FEFFFFFF0000000000000000010000000200000005000000FDFFFFFF0100000000000000010000000100000000000000"> : tensor<5x6x7xi32>
    %c_0 = stablehlo.constant dense<[[0, -4, 0, 0, -1, 0, 4], [0, 3, 1, 0, 0, -3, 4]]> : tensor<2x7xi32>
    return %c, %c_0 : tensor<5x6x7xi32>, tensor<2x7xi32>
  }
  func.func private @expected() -> (tensor<5x6x7xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0200000002000000FCFFFFFF0100000000000000020000000000000001000000FDFFFFFF000000000000000000000000010000000400000001000000F8FFFFFF030000000200000000000000FFFFFFFF02000000FFFFFFFF04000000010000000000000002000000FEFFFFFFFDFFFFFFFEFFFFFFFEFFFFFF01000000FDFFFFFF00000000FDFFFFFF01000000000000000000000002000000010000000700000002000000010000000300000003000000FFFFFFFF00000000010000000000000003000000FEFFFFFFFDFFFFFF000000000100000002000000FDFFFFFF00000000FFFFFFFFFFFFFFFF0100000005000000FDFFFFFF01000000FDFFFFFFFFFFFFFF000000000300000002000000030000000000000003000000020000000100000007000000F7FFFFFF0300000003000000FEFFFFFF01000000FDFFFFFF0400000004000000FEFFFFFFFDFFFFFF02000000FDFFFFFF0000000001000000FFFFFFFFFFFFFFFF040000000000000000000000FFFFFFFF000000000000000005000000FEFFFFFF0100000001000000FDFFFFFF01000000FFFFFFFF01000000F8FFFFFF010000000000000003000000010000000100000000000000FEFFFFFF040000000000000001000000FEFFFFFF02000000FCFFFFFF00000000FEFFFFFF020000000300000000000000020000000000000005000000FFFFFFFF02000000FFFFFFFF03000000FEFFFFFF00000000FDFFFFFF040000000400000000000000000000000200000003000000FDFFFFFF020000000100000003000000FFFFFFFF01000000FDFFFFFFFDFFFFFF0600000000000000FEFFFFFF03000000FFFFFFFF04000000FDFFFFFFFEFFFFFFF8FFFFFF0300000001000000FAFFFFFF030000000000000000000000FFFFFFFFFCFFFFFF000000000000000000000000FFFFFFFF0000000001000000000000000400000000000000FBFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFAFFFFFFFCFFFFFFFCFFFFFF000000000000000004000000FEFFFFFF000000000200000000000000FDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF01000000FFFFFFFFFEFFFFFF00000000FFFFFFFF00000000FEFFFFFF0000000000000000010000000200000005000000FDFFFFFF0100000000000000010000000100000000000000"> : tensor<5x6x7xi32>
    return %c : tensor<5x6x7xi32>
  }
}