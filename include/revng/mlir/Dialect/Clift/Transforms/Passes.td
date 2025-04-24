//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_PASSES
#define MLIR_CLIFT_PASSES

include "mlir/Pass/PassBase.td"

class Clift_Pass<string arg, string operation> : Pass<arg, operation> {
  let dependentDialects = ["clift::CliftDialect"];
}

def CliftModelAnalysis : Clift_Pass<"model", "mlir::ModuleOp"> {
  let constructor = "mlir::clift::createModelAnalysis()";

  let options = [
    Option<"Path", "path", "std::string", /*default*/"\"\"", "Model YAML path">,
  ];
}

def CliftImportModel : Clift_Pass<"import-model", "mlir::ModuleOp"> {
  let summary = "Import model types and add references to them in clift.undef ops";
  let constructor = "mlir::clift::createImportModelPass()";
}

def CliftVerifyC : Clift_Pass<"verify-c", "mlir::ModuleOp"> {
  let summary = "Verify that the Clift semantics correspond to C semantics " #
                "in the specified target implementation.";
  let constructor = "mlir::clift::createVerifyCPass()";
  let dependentDialects = ["clift::CliftDialect"];
}

def CliftEmitC : Pass<"emit-c", "mlir::ModuleOp"> {
  let summary = "Emit C code (optionally with PTML tags) from the Clift code.";
  let constructor = "mlir::clift::createEmitCPass()";
  let dependentDialects = ["clift::CliftDialect"];

  let options = [
    Option<"Tagless", "tagless", "bool", /*default=*/"false", "Emit without PTML tags">,
    Option<"Output", "output", "std::string", /*default*/"\"-\"", "C output path">
  ];
}

#endif // MLIR_CLIFT_PASSES
