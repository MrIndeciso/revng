//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <mutex>

#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"

#include "revng/ADT/SharedOnceFlag.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"
#include "revng/mlir/Dialect/Clift/Transforms/ModelAnalysis.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng/mlir/Dialect/Clift/Utils/CBackend.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"
#include "revng/Support/Debug.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTEMITC
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

struct EmitCPass : clift::impl::CliftEmitCBase<EmitCPass> {
  static std::unique_ptr<llvm::ToolOutputFile>
  tryOpenOutputFile(llvm::StringRef Filename) {
    std::string ErrorMessage;
    auto File = mlir::openOutputFile(Filename, &ErrorMessage);

    if (File)
      File->keep();
    else
      dbg << ErrorMessage << "\n";

    return File;
  }

  void runOnOperation() override {
    if (RunFlag.testAndSet()) {
      dbg << "emit-c cannot be used on inputs containing multiple modules.";

      return signalPassFailure();
    }

    auto File = tryOpenOutputFile(Output);
    if (not File)
      return signalPassFailure();

    const model::Binary *Model = clift::getModel(getPassState());
    if (Model == nullptr)
      return signalPassFailure();

    clift::TargetCImplementation Target = {
      .PointerSize = 8,
      .IntegerTypes = {
        { 1, clift::CIntegerKind::Char },
        { 2, clift::CIntegerKind::Short },
        { 4, clift::CIntegerKind::Int },
        { 8, clift::CIntegerKind::Long },
      },
    };

    llvm::raw_null_ostream NullStream;
    ptml::CTypeBuilder B(NullStream, *Model, /* EnableTaglessMode = */ Tagless);
    B.collectInlinableTypes();

    getOperation()->walk([&](clift::FunctionOp Function) {
      if (not Function.isExternal())
        File->os() << clift::decompile(Function, Target, B) << '\n';
    });
  }

  SharedOnceFlag RunFlag;
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> clift::createEmitCPass() {
  return std::make_unique<EmitCPass>();
}
