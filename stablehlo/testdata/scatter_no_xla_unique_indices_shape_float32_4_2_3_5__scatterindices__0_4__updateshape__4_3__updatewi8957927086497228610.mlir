// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xf32>, tensor<4x3xf32>)
    %2 = call @expected() : () -> tensor<4x2x3x5xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      stablehlo.return %arg1 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf32>, tensor<2xi32>, tensor<4x3xf32>) -> tensor<4x2x3x5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf32>, tensor<4x2x3x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf32>, tensor<4x3xf32>) {
    %0 = stablehlo.constant dense<"0x006680401BF99340B4B73BC0090D6DC081D48F3F36F630C02C349CC0FBD0454061D89DBF5068CABFFAE4A9BFBAF655C073575BBDC71A87BF5B2EFCBF75164FC0F76AAEBFDB2212C050EA1FC0AE29A9C096C18D3FA6A0C83F519A9040802927409B49F93F5492A33E7F8D0FC1FC0599403864A53FF7286DC0735FAEBF610BA0BFC1B57CBEC3C35FC028FBBF3EDAB605C03174513D0B558CBF781C303FE026F03FF5663AC029FB303F52348C3FDC6C2B40EB679C3FA35A0541D80108C0CD5DF5BF063B2EBE3F88E33FC5E6923FFDBB96BF2D3608BFE69B8ABF5955983BE0A4CABFB8A0F9C0E5B71D40F2D2024075E658C0C6832D3F7410483F547B5A3F99020241B380B23F9B89AF3E99284240C82D00C00925A7C0AE4D89BFCE22394087F70E4079838A405F3C4740BBB54C40E36AF5BEDF7F9DC0F74EC940786B35BFD0A5EA3DDF99923FB7CF943FF5625CC05BD333C0E3B791BF684B3E3F762E9040FEA5C8BF568F9AC095354DBE6525B7BFBF359BBF9127BFBF2B1A91BF8AD147C035E0B3BEE2BA90C08C931340ABE25C40EBF5A84050609F3EE34B9440C418163DFA2D27C0FE3BA5BF42525A3FBF9FA3C0616355BF7DA13D403F4BA0C041120DBF2914B0BFF7842540BAC30A40F18ED3BF0EE76640C44A3640C9CB673E2953A1BF140C55C0"> : tensor<4x2x3x5xf32>
    %1 = stablehlo.constant dense<[[-2.2441895, -2.47734118, -2.79215312], [-5.956429, 0.419247031, 3.8496623], [1.71141255, -1.50947154, 0.377479464], [4.30061865, -6.32353449, 1.80235553]]> : tensor<4x3xf32>
    return %0, %1 : tensor<4x2x3x5xf32>, tensor<4x3xf32>
  }
  func.func private @expected() -> tensor<4x2x3x5xf32> {
    %0 = stablehlo.constant dense<"0x006680401BF99340B4B73BC0090D6DC0CDA00FC036F630C02C349CC0FBD0454061D89DBFC28C1EC0FAE4A9BFBAF655C073575BBDC71A87BFA3B232C075164FC0F76AAEBFDB2212C050EA1FC0AE29A9C096C18D3FA6A0C83F519A9040802927409B49F93F5492A33E7F8D0FC1FC0599403864A53FF7286DC0735FAEBF610BA0BFC1B57CBEC3C35FC0119BBEC0DAB605C03174513D0B558CBF781C303F8CA7D63EF5663AC029FB303F52348C3FDC6C2B40DE607640A35A0541D80108C0CD5DF5BF063B2EBE3F88E33FC5E6923FFDBB96BF2D3608BFE69B8ABF5955983BE0A4CABFB8A0F9C0E5B71D40F2D2024075E658C0C6832D3F7410483F547B5A3F99020241910FDB3F9B89AF3E99284240C82D00C00925A7C05D36C1BFCE22394087F70E4079838A405F3C4740FD44C13EE36AF5BEDF7F9DC0F74EC940786B35BFD0A5EA3DDF99923FB7CF943FF5625CC05BD333C0E3B791BF684B3E3F762E9040FEA5C8BF568F9AC095354DBE6525B7BFBF359BBF9127BFBF2B1A91BFAB9E894035E0B3BEE2BA90C08C931340ABE25C40655ACAC050609F3EE34B9440C418163DFA2D27C096B3E63F42525A3FBF9FA3C0616355BF7DA13D403F4BA0C041120DBF2914B0BFF7842540BAC30A40F18ED3BF0EE76640C44A3640C9CB673E2953A1BF140C55C0"> : tensor<4x2x3x5xf32>
    return %0 : tensor<4x2x3x5xf32>
  }
}
