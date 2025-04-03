//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IRReader/IRReader.h"

#include "mlir/Pass/Pass.h"

#include "revng/mlir/Dialect/Clift/Transforms/ModelAnalysis.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportLLVM.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTIMPORTLLVM
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

struct ImportLLVMPass : clift::impl::CliftImportLLVMBase<ImportLLVMPass> {
  void runOnOperation() override {
    mlir::ModuleOp Module = getOperation();

    if (not Module.getBody()->empty())
      return signalPassFailure();

    if (not Module->getAttrs().empty())
      return signalPassFailure();

    const model::Binary *Model = clift::getModel(getPassState());
    if (Model == nullptr)
      return signalPassFailure();

    llvm::LLVMContext LLVMContext;

    llvm::SMDiagnostic Diag;
    auto LLVMModule = llvm::parseIRFile(LLVMIRPath, Diag, LLVMContext);
    if (LLVMModule == nullptr) {
      Diag.print("", llvm::errs());
      return signalPassFailure();
    }

    clift::setModuleAttr(Module);
    clift::importLLVM(Module, *Model, LLVMModule.get());
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
clift::createImportLLVMPass() {
  return std::make_unique<ImportLLVMPass>();
}
