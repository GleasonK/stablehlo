/* Copyright 2022 The StableHLO Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "stablehlo/transforms/TypeConversion.h"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "stablehlo/dialect/VhloOps.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace vhlo {

void registerFuncOpsForTypeConversion(ConversionTarget& target,
                                      RewritePatternSet& patterns,
                                      TypeConverter& converter) {
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return converter.isSignatureLegal(op.getCalleeType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return converter.isLegal(op.getOperandTypes());
  });
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);
  populateReturnOpTypeConversionPattern(patterns, converter);
}

namespace {
/// Adds either an upgrade conversion or a downgrade conversion depending on the
/// specified target version.
///
/// If target is less than or equal to the downgraded type, adds the downgrade
/// conversion. Else adds the upgrade conversion.
template <typename DowngradedType, typename UpgradedType>
void addUpgradeOrDowngrade(
    TypeConverter& converter, Version const& target,
    std::function<UpgradedType(DowngradedType)>&& upgradeFn,
    std::function<DowngradedType(UpgradedType)>&& downgradeFn) {
  assert(DowngradedType::maxVersion() < UpgradedType::minVersion());
  if (target <= DowngradedType::maxVersion()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Adding type conversion " << UpgradedType::getMnemonic()
               << " -> " << DowngradedType::getMnemonic() << '\n');
    converter.addConversion(downgradeFn);
  } else {
    assert(UpgradedType::maxVersion() <= target);
    LLVM_DEBUG(llvm::dbgs()
               << "Adding type conversion " << DowngradedType::getMnemonic()
               << " -> " << UpgradedType::getMnemonic() << '\n');
    converter.addConversion(upgradeFn);
  }
}
}  // namespace

VhloToVersionConverter::VhloToVersionConverter(Version const& target)
    : VersionedTypeConverterBase() {
  // Base conversion - Allow types that are legal in the target version.
  addConversion([&](Type type) -> Type {
    if (auto interface = type.dyn_cast<VersionedTypeInterface>()) {
      LLVM_DEBUG(llvm::dbgs() << "Checking type legality " << type << ": "
                              << interface.getMinVersion() << " <= " << target
                              << " <= " << interface.getMaxVersion() << '\n');
      return isLegalVersionForTarget(interface, target) ? type : Type{};
    }
    // TODO: All types should be versioned
    return type;
  });

  // TokenType <--> TokenV2Type
  addUpgradeOrDowngrade<TokenType, TokenV2Type>(
      *this, target,
      [](TokenType type) { return TokenV2Type::get(type.getContext()); },
      [](TokenV2Type type) { return TokenType::get(type.getContext()); });
}

bool VhloToVersionConverter::isSourceDialect(Dialect& dialect) {
  return dialect.getNamespace() == vhlo::VhloDialect::getDialectNamespace();
}

Attribute VhloToVersionConverter::convertEncoding(Attribute attr) {
  return attr;
}

}  // namespace vhlo
}  // namespace mlir
