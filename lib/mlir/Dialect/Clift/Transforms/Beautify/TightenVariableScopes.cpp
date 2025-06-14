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

static llvm::SmallDenseMap<clift::LocalVariableOp, mlir::Operation *>
getFirstPostOrderUsers(llvm::SmallVector<clift::LocalVariableOp> LocalVariables,
                       mlir::Region *region) {
  llvm::SmallDenseMap<clift::LocalVariableOp, mlir::Operation *> FirstUsers;

  // Walk through the region and collect the first user of each local variable
  // This must be done post-order, as want the first element of the users
  // vector to be the first possible user of the value
  // Otherwise, findCommonAncestorOp will not find the earlist common operation
  region->walk([&](mlir::Operation *op) {
    for (clift::LocalVariableOp LocalVarOp : LocalVariables) {
      if (FirstUsers.count(LocalVarOp) > 0) {
        // If we already have a first user, we do not need to check this op
        continue;
      }

      for (mlir::Value user : op->getOperands()) {
        if (user == LocalVarOp.getResult()) {
          FirstUsers[LocalVarOp] = op;
          break;
        }
      }
    }

    if (FirstUsers.size() == LocalVariables.size()) {
      // If we have found the first user for all local variables, we can stop
      // walking the region
      return mlir::WalkResult::interrupt();
    } else {
      // Continue walking the region
      return mlir::WalkResult::advance();
    }
  });

  return FirstUsers;
}

static mlir::Operation *findCommonAncestorOp(mlir::Operation *FirstUser,
                                             mlir::Value::user_range Users) {
  if (Users.empty()) {
    return nullptr;
  }

  // We are looking for the common ancestor operation, after which all uses
  // happen. To do so, we iterate over all the parent operations of the first
  // operation until that operation's parent region is an ancestor of all the
  // operations
  mlir::Operation *CommonFirstOp = FirstUser;
  mlir::Region *CommonParentRegion = CommonFirstOp->getParentRegion();

  while (true) {
    bool AllInSameParent = true;
    for (mlir::Operation *Op : Users) {
      if (not CommonParentRegion->isProperAncestor(Op->getParentRegion())) {
        AllInSameParent = false;
        break;
      }
    }
    if (AllInSameParent) {
      break;
    }

    CommonFirstOp = CommonFirstOp->getParentOp();
    CommonParentRegion = CommonFirstOp->getParentRegion();
  }

  return CommonFirstOp;
}

struct TightenVariableScopePass
    : clift::impl::CliftTightenVariableScopesBase<TightenVariableScopePass> {
  void runOnOperation() override {
    clift::FunctionOp FunctionOp = getOperation();

    // Check if the function has a body
    if (FunctionOp.getBody().empty()) {
      // Nothing to do
      return;
    }

    // Store the function's local variables in a vector
    llvm::SmallVector<clift::LocalVariableOp> LocalVariables;

    // Parse all the local variables of the function
    for (mlir::Operation &Op : FunctionOp.getBody().getOps()) {
      if (auto LocalVarOp = mlir::dyn_cast<clift::LocalVariableOp>(Op)) {
        // Local variables with no users are ignored
        // TODO: this would probably never happen, as other passes should
        // eliminate them, so maybe remove this check?
        if (LocalVarOp.getResult().use_empty()) {
          continue;
        }

        // Local variables with initializers must not be moved
        if (not LocalVarOp.getInitializer().empty()) {
          continue;
        }

        LocalVariables.push_back(LocalVarOp);
      }
      //  else {
      // ASSUMPTION: all the local variables are always at the start of the
      // function
      // Thinking about this, we might be able to break once we encounter an op
      // with subregions maybe "else if (op.getNumRegions()) break;" break;
      // }
    }

    // If there are no local variables, nothing to do
    if (LocalVariables.empty()) {
      return;
    }

    auto FirstUsers =
        getFirstPostOrderUsers(LocalVariables, &FunctionOp.getBody());

    for (clift::LocalVariableOp LocalVarOp : LocalVariables) {
      // Find the outermost region containing all the users of the local
      // variable
      auto Users = LocalVarOp.getResult().getUsers();

      // We attempt to find the outermost first op which contains a use of this
      // local
      mlir::Operation *AncestorOp =
          findCommonAncestorOp(FirstUsers[LocalVarOp], Users);

      // ancestorOp might be the successor of localVarOp
      // In this case, our rewrite would effectively be a no-op
      if (AncestorOp->getPrevNode() == LocalVarOp.getOperation()) {
        continue;
      }

      // Insert a new operation in the location we identified
      LocalVarOp->moveBefore(AncestorOp);
    }
  }
};

} // namespace

clift::PassPtr<clift::FunctionOp> clift::createTightenVariableScopePass() {
  return std::make_unique<TightenVariableScopePass>();
}
