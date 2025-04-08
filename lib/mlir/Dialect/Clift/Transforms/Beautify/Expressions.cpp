//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Transforms/Rewrites.h"

namespace clift = mlir::clift;

static uint64_t truncateIntegerValue(mlir::IntegerAttr ValueAttr,
                                     mlir::Value IntegerOperand) {
  auto ValueType = mlir::cast<clift::ValueType>(IntegerOperand.getType());
  auto T = mlir::cast<clift::PrimitiveType>(clift::dealias(ValueType, true));

  uint64_t Value = ValueAttr.getValue().getZExtValue();
  return Value & (static_cast<uint64_t>(-1) >> (64 - 8 * T.getSize()));
}

static bool isCollapsibleCastKind(clift::CastKind Kind) {
  return Kind != clift::CastKind::Convert;
}

static bool hasEnumeratorValue(clift::ValueType Type, uint64_t Value) {
  if (auto Enum = mlir::dyn_cast<clift::EnumType>(Type)) {
    for (clift::EnumFieldAttr Enumerator : Enum.getFields()) {
      if (Enumerator.getRawValue() == Value)
        return true;
    }
  }
  return false;
}

struct DivModPair {
  uint64_t Div;
  uint64_t Mod;
};

static DivModPair ptrOffsetDivMod(mlir::IntegerAttr OffsetAttr,
                                  mlir::Value PointerOperand) {
  auto PointerType = mlir::cast<clift::PointerType>(PointerOperand.getType());

  uint64_t Offset = OffsetAttr.getValue().getZExtValue();
  uint64_t Size = PointerType.getPointeeType().getByteSize();

  if (Size == 0) {
    return {
      .Div = 0,
      .Mod = static_cast<uint64_t>(-1),
    };
  }

  return {
    .Div = Offset / Size,
    .Mod = Offset % Size,
  };
}

namespace {
#include "revng/mlir/Dialect/Clift/Transforms/Beautify/Expressions.h.inc"
} // namespace

void clift::populateBeautifyExpressionRewritePatterns(RewritePatternSet
                                                        &Patterns) {
  populateWithGenerated(Patterns);
}
