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
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    clift::SwitchOp switchOp = mlir::cast<clift::SwitchOp>(op);

    // Reject if there are more than one case
    if (switchOp.getNumCases() != 1) {
      return rewriter.notifyMatchFailure(
          switchOp, "Only one case supported, found: " +
                        std::to_string(switchOp.getNumCases()));
    }

    rewriter.setInsertionPoint(switchOp);

    // Create the ifOp at the position of the switchOp
    clift::IfOp ifOp = rewriter.create<clift::IfOp>(switchOp.getLoc());

    // The ifOp inherits the regions of the switchOp
    ifOp.getCondition().takeBody(switchOp.getCondition());
    ifOp.getThen().takeBody(switchOp.getCaseRegion(0));
    ifOp.getElse().takeBody(switchOp.getDefaultCaseRegion());

    uint64_t caseValue = switchOp.getCaseValue(0);

    // Erase the switchOp
    rewriter.eraseOp(switchOp);

    // Erase the last of the then block, if it is a switch break
    // ASSUMPTION: regions only contain a single block
    if (mlir::isa<clift::SwitchBreakOp>(ifOp.getThen().front().back())) {
      // rewriter.setInsertionPointToEnd(&ifOp.getThen().front());
      rewriter.eraseOp(&ifOp.getThen().front().back());
    }

    // Erase the last op of the else block (switch break)
    // ASSUMPTION: regions only contain a single block
    if (mlir::isa<clift::SwitchBreakOp>(ifOp.getElse().front().back())) {
      // rewriter.setInsertionPointToEnd(&ifOp.getElse().front());
      rewriter.eraseOp(&ifOp.getElse().front().back());
    }

    // Find the yield op in the condition block
    clift::YieldOp yieldOp = clift::getExpressionYieldOp(ifOp.getCondition());

    if (not yieldOp)
      return rewriter.notifyMatchFailure(
          switchOp, "Condition block must end with a yield op");

    // We insert the new ops just before the yield op
    rewriter.setInsertionPoint(yieldOp.getOperation());

    // Create the immediate op for the case value
    clift::ImmediateOp caseValueOp = rewriter.create<clift::ImmediateOp>(
        ifOp.getLoc(), yieldOp.getOperand().getType(), caseValue);

    // Get the yield op operand
    mlir::Value previousYieldOp = yieldOp.getOperand();

    // Create a comparison op
    clift::CmpEqOp cmpOp = rewriter.create<clift::CmpEqOp>(
        yieldOp.getLoc(),
        clift::PrimitiveType::get(switchOp.getContext(),
                                  clift::PrimitiveKind::SignedKind, 1, false),
        previousYieldOp, caseValueOp.getResult());

    // Yield the result of the comparison
    yieldOp.setOperand(cmpOp.getResult());

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

std::unique_ptr<mlir::OperationPass<clift::ModuleOp>>
clift::createSwitchCaseRewritePass() {
  return std::make_unique<SwitchCaseRewritePass>();
}
