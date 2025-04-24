//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"

#include "revng/mlir/Dialect/Clift/Transforms/ModelAnalysis.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng/Support/Debug.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTMODELANALYSIS
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

using TupleTreeType = TupleTree<model::Binary>;

struct ModelAnalysisPass
  : clift::impl::CliftModelAnalysisBase<ModelAnalysisPass> {

  void runOnOperation() {
    auto MaybeModel = TupleTreeType::fromFile(Path);

    if (not MaybeModel) {
      dbg << "Failed to parse model: " << consumeToString(MaybeModel) << '\n';
      return signalPassFailure();
    }

    if (not MaybeModel->verify()) {
      dbg << "Model verification failed\n";
      return signalPassFailure();
    }

    auto P = makeTypeErased<TupleTreeType>(std::move(*MaybeModel));
    const auto *Model = static_cast<const TupleTreeType *>(P.get())->get();

    auto &Analysis = clift::getModelAnalysis(getPassState());
    Analysis.Binary = TypeErasedPtr<const model::Binary>(std::move(P), Model);
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
clift::createModelAnalysis() {
  return std::make_unique<ModelAnalysisPass>();
}
