//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_PASSES
#define MLIR_CLIFT_PASSES

include "mlir/Pass/PassBase.td"

class Clift_Pass<string arg, string operation> : Pass<arg, operation> {
  let dependentDialects = ["clift::CliftDialect"];
}

def CliftImportLLVM : Clift_Pass<"import-llvm", "mlir::ModuleOp"> {
  let summary = "";
  let constructor = "mlir::clift::createImportLLVMPass()";

  let options = [
    Option<"LLVMIRPath", "llvm", "std::string", /*default*/"\"\"", "LLVM IR Path">,
    Option<"Model", "model", "mlir::clift::ModelOptionType", /*default=*/"nullptr", "Model path">
  ];
}

// TODO: Act on clift::FunctionOp
def CliftLoopDetection : Clift_Pass<"loop-detection", "clift::ModuleOp"> {
  let summary = "Detect Clift loops";
  let constructor = "mlir::clift::createLoopDetectionPass()";
}

// TODO: Act on clift::FunctionOp
def CliftBranchEqualization : Clift_Pass<"branch-equalization",
                                         "clift::ModuleOp"> {
  let summary = "Equalize Clift branches";
  let constructor = "mlir::clift::createBranchEqualizationPass()";
}

// TODO: Act on clift::FunctionOp
def CliftBeautify : Clift_Pass<"beautify", "clift::ModuleOp"> {
  let summary = "Beautify Clift";
  let constructor = "mlir::clift::createBeautifyPass()";
}

def CliftVerifyC : Clift_Pass<"verify-c", "clift::ModuleOp"> {
  let summary = "Verify that the Clift semantics correspond to C semantics " #
                "in the specified target implementation.";
  let constructor = "mlir::clift::createVerifyCPass()";
}

def CliftEmitC : Clift_Pass<"emit-c", "clift::ModuleOp"> {
  let summary = "Emit C code (optionally with PTML tags) from the Clift code.";
  let constructor = "mlir::clift::createEmitCPass()";

  let options = [
    Option<"Tagless", "tagless", "bool", /*default=*/"false", "Emit without PTML tags">,
    Option<"Model", "model", "mlir::clift::ModelOptionType", /*default=*/"nullptr", "Model path">,
    Option<"Output", "output", "std::string", /*default*/"\"-\"", "C output path">
  ];
}

#endif // MLIR_CLIFT_PASSES
