//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Transforms/ModelAnalysis.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTIMPORTMODEL
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

static void importAllModelTypes(mlir::ModuleOp Module,
                                const model::Binary &Model) {
  mlir::MLIRContext *const Context = Module->getContext();

  const auto EmitError = [&]() -> mlir::InFlightDiagnostic {
    return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                         mlir::DiagnosticSeverity::Error);
  };

  mlir::OpBuilder Builder(Module.getRegion());
  for (const auto &ModelType : Model.TypeDefinitions()) {
    auto Type = clift::importModelType(EmitError, *Context, *ModelType, Model);
    Builder.create<clift::UndefOp>(mlir::UnknownLoc::get(Context), Type);
  }
}

using clift::impl::CliftImportModelBase;
struct ImportModelPass : CliftImportModelBase<ImportModelPass> {
  void runOnOperation() override {
    const model::Binary *Model = clift::getModel(getPassState());
    if (Model == nullptr)
      return signalPassFailure();

    importAllModelTypes(getOperation(), *Model);
  };
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
clift::createImportModelPass() {
  return std::make_unique<ImportModelPass>();
}
