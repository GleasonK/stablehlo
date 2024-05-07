// RUN: stablehlo-opt --mlir-print-op-generic %s.bc | FileCheck %s
// RUN: stablehlo-translate --deserialize %s.bc | stablehlo-translate --serialize --target=0.19.0 | stablehlo-translate --deserialize --mlir-print-op-generic | FileCheck --check-prefix=DESERIALIZED %s

// This file is bytecoded targeting 0.19.0 for ops that are removed in 0.20.0
// ensuring that the bytecode files are still readable.
// If regenerating bytecode file is needed, must rebuild at 0.19.0 since this
// file's MLIR textual assembly is invalid at HEAD.

// CHECK-LABEL: "op_create_token"
func.func @op_create_token() -> !stablehlo.token {
  // CHECK: "vhlo.create_token_v1"() : () -> !vhlo.token_v1
  // DESERIALIZED: "stablehlo.after_all"() : () -> !stablehlo.token
  %0 = "stablehlo.create_token"() : () -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
