/* Copyright 2025 The OpenXLA Authors.

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

#ifndef STABLEHLO_BUILDER_ATTRTYPEBUILDERUTIL_H_
#define STABLEHLO_BUILDER_ATTRTYPEBUILDERUTIL_H_

#include <complex>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <type_traits>
#include <vector>

#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/TypeSwitch.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Builders.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinAttributes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinTypeInterfaces.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinTypes.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Types.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/LLVM.h"

namespace mlir {

// POD type to Tensor element type map

// List of supported Tensor Element Types.
// This list is fairly XLA specific, used to provide sugar for the common
// RankedTensorType's we'll need to build.
// TODO: Add more element types.
enum class ElementType {
  // clang-format off
  PRED,
  I2, I4, I8, I16, I32, I64,
  UI2, UI4, UI8, UI16, UI32, UI64,
  BF16, F16, F32, F64,
  F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
  F8E4M3FN, F8E4M3FNUZ, F8E4M3B11FNUZ, F8E5M2, F8E5M2FNUZ, F8E8M0FNU,
  COMPLEXF32, COMPLEXF64
  // clang-format on
};

Type getElementType(MLIRContext& ctx, ElementType elementType);

// clang-format off
// template <> struct type_to_element_type<int8_t> { using Type = ElementType::I8; };
// template <> struct type_to_element_type<int16_t> { using Type = ElementType::I16 };
// template <> struct type_to_element_type<int32_t> { using Type = ElementType::I32 };
// template <> struct type_to_element_type<uint8_t, ElementType::UI8>;
// template <> struct type_to_element_type<uint16_t, ElementType::UI16>;
// template <> struct type_to_element_type<uint32_t, ElementType::UI32>;
// template <> struct type_to_element_type<uint64_t, ElementType::UI64>;
// template <> struct type_to_element_type<float, ElementType::F32>;
// template <> struct type_to_element_type<double, ElementType::F64>;
// clang-format on

// ElementType enum to MLIR Type

RankedTensorType makeTensorType(MLIRContext& ctx, ArrayRef<int64_t> shape,
                                ElementType elementType);

APFloat toAPFloat(double val, FloatType floatType);

namespace detail {

template <typename T>
struct type_to_element_type;

template <>
struct type_to_element_type<int64_t> {
  static constexpr ElementType kType = ElementType::I64;
};

template <typename>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

// Helper for handling integer numbers
template <typename T>
DenseElementsAttr makeConstantInt(RankedTensorType tensorType,
                                  ArrayRef<T> values) {
  auto intType = dyn_cast<IntegerType>(tensorType.getElementType());
  if (!intType) return nullptr;

  // This branch is for integer types whose data width is <= 8 bits.
  if (intType.getWidth() <= 8) {
    SmallVector<char> packedPaddedData(values.begin(), values.end());
    return DenseElementsAttr::getFromRawBuffer(tensorType, packedPaddedData);
  }

  // This memcpy approach works for byte-aligned integer types (i8, i16,
  // i32, etc.)
  unsigned bitwidth = intType.getWidth();
  if (bitwidth % 8 == 0) {
    size_t elementSizeBytes = bitwidth / 8;
    std::vector<char> rawData(values.size() * elementSizeBytes);
    char* destination = rawData.data();
    for (const auto& value : values) {
      memcpy(destination, &value, elementSizeBytes);
      destination += elementSizeBytes;
    }
    return DenseElementsAttr::getFromRawBuffer(tensorType, rawData);
  }

  // Unsupported type.
  return nullptr;
}

// Helper for handling floating-point numbers
template <typename T>
DenseElementsAttr makeConstantFloat(RankedTensorType tensorType,
                                    ArrayRef<T> values) {
  // Creating complex types from float types is not supported for arrays.
  auto complexType = dyn_cast<ComplexType>(tensorType.getElementType());
  if (complexType) return nullptr;

  auto floatType = dyn_cast<FloatType>(tensorType.getElementType());
  if (!floatType) return nullptr;

  SmallVector<APFloat> floatValues;
  floatValues.reserve(values.size());

  for (const auto& value : values) {
    floatValues.push_back(toAPFloat(static_cast<double>(value), floatType));
  }
  return DenseElementsAttr::get(tensorType, floatValues);
}

// Helper for handling complex numbers
template <typename T>
DenseElementsAttr makeConstantComplex(RankedTensorType tensorType,
                                      ArrayRef<T> values) {
  auto complexType = dyn_cast<ComplexType>(tensorType.getElementType());
  if (!complexType) return nullptr;

  auto floatType = dyn_cast<FloatType>(complexType.getElementType());
  if (!floatType) return nullptr;

  SmallVector<std::complex<APFloat>> complexValues;
  complexValues.reserve(values.size());

  for (const auto& value : values) {
    complexValues.emplace_back(toAPFloat(value.real(), floatType),
                               toAPFloat(value.imag(), floatType));
  }
  return DenseElementsAttr::get(tensorType, complexValues);
}

}  // namespace detail

// Locations
Location unknownLoc(MLIRContext& ctx);
Location fileLineColLoc(MLIRContext& ctx, StringRef file, int64_t line,
                        int64_t col);
Location cppFileLineColLoc(
    MLIRContext& ctx,
    const std::source_location& loc = std::source_location::current());

template <typename T>
DenseElementsAttr makeConstant(MLIRContext& ctx, T value,
                               RankedTensorType tensorType) {
  return TypeSwitch<Type, DenseElementsAttr>(tensorType.getElementType())
      .template Case<IntegerType>([&](IntegerType type) -> DenseElementsAttr {
        if constexpr (std::is_integral_v<T>) {
          return DenseElementsAttr::get(tensorType,
                                        IntegerAttr::get(type, value));
        }
        return nullptr;
      })
      .template Case<FloatType>([&](FloatType type) -> DenseElementsAttr {
        if constexpr (std::is_arithmetic_v<T>) {
          return DenseElementsAttr::get(
              tensorType, FloatAttr::get(type, static_cast<double>(value)));
        }
        return nullptr;
      })
      .template Case<ComplexType>([&](ComplexType type) -> DenseElementsAttr {
        auto floatType = dyn_cast<FloatType>(type.getElementType());
        if (!floatType) return nullptr;

        if constexpr (detail::is_complex_v<T>) {
          return DenseElementsAttr::get(
              tensorType,
              std::complex<APFloat>(
                  toAPFloat(static_cast<double>(value.real()), floatType),
                  toAPFloat(value.imag(), floatType)));
        }

        if constexpr (std::is_arithmetic_v<T>) {
          return DenseElementsAttr::get(
              tensorType, std::complex<APFloat>(
                              toAPFloat(static_cast<double>(value), floatType),
                              toAPFloat(0.0, floatType)));
        }

        return nullptr;
      })
      .Default([](Type) -> DenseElementsAttr {
        // Unsupported type.
        return nullptr;
      });
}

template <typename T>
DenseElementsAttr makeConstant(MLIRContext& ctx, ArrayRef<T> values,
                               RankedTensorType tensorType) {
  if constexpr (std::is_same_v<T, bool>) {
    return DenseElementsAttr::get(tensorType, values);
  }

  if constexpr (std::is_integral_v<T>) {
    return makeConstantInt(tensorType, values);
  }

  if constexpr (std::is_floating_point_v<T>) {
    return makeConstantFloat(tensorType, values);
  }

  if constexpr (detail::is_complex_v<T>) {
    return makeConstantComplex(tensorType, values);
  }

  // Unsupported type.
  return nullptr;
}

template <typename T>
DenseElementsAttr makeConstant(MLIRContext& ctx, T value,
                               ArrayRef<int64_t> shape,
                               ElementType elementType) {
  static_assert(!std::is_same<T, int>::value,
                "This method doesn't work with raw int literals, use intN_t or "
                "suffixed literals like 1L instead.");
  return DenseElementsAttr::get(makeTensorType(ctx, shape, elementType), value);
}

}  // namespace mlir

#endif  // STABLEHLO_BUILDER_ATTRTYPEBUILDERUTIL_H_
