//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTTIGHTENVARIABLESCOPES
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

static llvm::SmallVector<mlir::Operation *>
getPostOrderUsers(mlir::Value value, mlir::Region *region) {
  llvm::SmallVector<mlir::Operation *> users;

  // Walk through the region and collect all the users of the value
  // This must be done post-order, as want the first element of the users
  // vector to be the first possible user of the value
  // Otherwise, findCommonAncestorOp will not find the earlist common operation
  region->walk([&](mlir::Operation *op) {
    for (mlir::Value user : op->getOperands()) {
      if (user == value) {
        users.push_back(op);
        break;
      }
    }
  });

  return users;
}

static mlir::Operation *
findCommonAncestorOp(llvm::SmallVector<mlir::Operation *> ops) {
  if (ops.empty()) {
    return nullptr;
  }

  // We are looking for the common ancestor operation, after which all uses
  // happen. To do so, we iterate over all the parent operations of the first
  // operation until that operation's parent region is an ancestor of all the
  // operations
  mlir::Operation *commonFirstOp = *ops.begin();
  mlir::Region *commonParentRegion = commonFirstOp->getParentRegion();

  while (true) {
    bool allInSameParent = true;
    for (mlir::Operation *op : ops) {
      if (not commonParentRegion->isProperAncestor(op->getParentRegion())) {
        allInSameParent = false;
        break;
      }
    }
    if (allInSameParent) {
      break;
    }

    commonFirstOp = commonFirstOp->getParentOp();
    commonParentRegion = commonFirstOp->getParentRegion();
  }

  return commonFirstOp;
}

struct TightenVariableScopePattern : mlir::RewritePattern {
  TightenVariableScopePattern(mlir::MLIRContext *Context)
      : mlir::RewritePattern(clift::FunctionOp::getOperationName(), 1,
                             Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    clift::FunctionOp functionOp = mlir::cast<clift::FunctionOp>(op);

    // Check if the function has a body
    if (functionOp.getBody().empty()) {
      return rewriter.notifyMatchFailure(functionOp, "Function has no body");
    }

    // Store the function's local variables in a vector
    llvm::SmallVector<clift::LocalVariableOp> localVariables;

    // Parse all the local variables of the function
    for (mlir::Operation &op : functionOp.getBody().getOps()) {
      if (auto localVarOp = mlir::dyn_cast<clift::LocalVariableOp>(op)) {
        // Local variables with no users are ignored
        // TODO: this would probably never happen, as other passes should
        // eliminate them, so maybe remove this check?
        if (localVarOp.getResult().use_empty()) {
          continue;
        }

        // Local variables with initializers must not be moved
        if (not localVarOp.getInitializer().empty()) {
          continue;
        }

        localVariables.push_back(localVarOp);
      }
      //  else {
      // ASSUMPTION: all the local variables are always at the start of the
      // function
      // Thinking about this, we might be able to break once we encounter an op
      // with subregions maybe "else if (op.getNumRegions()) break;" break;
      // }
    }

    // If there are no local variables, nothing to do
    if (localVariables.empty()) {
      return rewriter.notifyMatchFailure(functionOp,
                                         "No local variables to move");
    }

    // The pass succeeds if at least one local variable is moved
    bool atLeastOneMoved = false;

    for (clift::LocalVariableOp localVarOp : localVariables) {
      // Find the outermost region containing all the users of the local
      // variable
      auto users =
          getPostOrderUsers(localVarOp.getResult(), &functionOp.getBody());

      // We attempt to find the outermost first op which contains a use of this
      // local
      mlir::Operation *ancestorOp = findCommonAncestorOp(users);

      // ancestorOp might be the successor of localVarOp
      // In this case, our rewrite would effectively be a no-op
      if (ancestorOp->getPrevNode() == localVarOp.getOperation()) {
        continue;
      }

      // Move the local variable right before the use op
      rewriter.setInsertionPoint(ancestorOp);

      // Insert a new operation in the location we identified
      mlir::Operation *new_op = rewriter.clone(*localVarOp.getOperation());
      rewriter.replaceOp(localVarOp, new_op->getResults());

      // Record that at least one local variable was moved
      atLeastOneMoved = true;
    }

    if (not atLeastOneMoved) {
      return rewriter.notifyMatchFailure(functionOp,
                                         "No local variables could be moved");
    }

    return mlir::success();
  }
};

struct TightenVariableScopePass
    : clift::impl::CliftTightenVariableScopesBase<TightenVariableScopePass> {
  void runOnOperation() override {
    mlir::MLIRContext &Context = getContext();

    mlir::RewritePatternSet Patterns(&Context);

    Patterns.add<TightenVariableScopePattern>(&Context);

    if (mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<clift::ModuleOp>>
clift::createTightenVariableScopePass() {
  return std::make_unique<TightenVariableScopePass>();
}
