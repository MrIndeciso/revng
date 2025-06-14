//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTSWITCHCASEREWRITE
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

struct SwitchCaseRewritePattern : mlir::RewritePattern {
  SwitchCaseRewritePattern(mlir::MLIRContext *Context)
      : mlir::RewritePattern(clift::SwitchOp::getOperationName(), 1, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &Rewriter) const override {
    clift::SwitchOp SwitchOp = mlir::cast<clift::SwitchOp>(Op);

    // Reject if there is more than one case
    if (SwitchOp.getNumCases() != 1) {
      return Rewriter.notifyMatchFailure(
          SwitchOp, "Only one case supported, found: " +
                        std::to_string(SwitchOp.getNumCases()));
    }

    Rewriter.setInsertionPoint(SwitchOp);

    // Create the IfOp at the position of the switchOp
    clift::IfOp IfOp = Rewriter.create<clift::IfOp>(SwitchOp.getLoc());

    // The ifOp inherits the regions of the switchOp
    IfOp.getCondition().takeBody(SwitchOp.getCondition());
    IfOp.getThen().takeBody(SwitchOp.getCaseRegion(0));
    IfOp.getElse().takeBody(SwitchOp.getDefaultCaseRegion());

    uint64_t CaseValue = SwitchOp.getCaseValue(0);

    // Erase the switchOp
    Rewriter.eraseOp(SwitchOp);

    // Find the yield op in the condition block
    clift::YieldOp YieldOp = clift::getExpressionYieldOp(IfOp.getCondition());

    if (not YieldOp)
      return Rewriter.notifyMatchFailure(
          SwitchOp, "Condition block must end with a yield op");

    // We insert the new ops just before the yield op
    Rewriter.setInsertionPoint(YieldOp.getOperation());

    // Create the immediate op for the case value
    clift::ImmediateOp CaseValueOp = Rewriter.create<clift::ImmediateOp>(
        IfOp.getLoc(), YieldOp.getOperand().getType(), CaseValue);

    // Get the original yield op operand
    mlir::Value PreviousYieldOp = YieldOp.getOperand();

    // Create a comparison op
    clift::CmpEqOp CmpOp = Rewriter.create<clift::CmpEqOp>(
        YieldOp.getLoc(),
        clift::PrimitiveType::get(SwitchOp.getContext(),
                                  clift::PrimitiveKind::SignedKind, 4, false),
        PreviousYieldOp, CaseValueOp.getResult());

    // Yield the result of the comparison
    YieldOp.setOperand(CmpOp.getResult());

    return mlir::success();
  }
};

struct SwitchCaseRewritePass
    : clift::impl::CliftSwitchCaseRewriteBase<SwitchCaseRewritePass> {
  void runOnOperation() override {
    mlir::MLIRContext &Context = getContext();

    mlir::RewritePatternSet Patterns(&Context);

    Patterns.add<SwitchCaseRewritePattern>(&Context);

    if (mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace

clift::PassPtr<clift::FunctionOp> clift::createSwitchCaseRewritePass() {
  return std::make_unique<SwitchCaseRewritePass>();
}
