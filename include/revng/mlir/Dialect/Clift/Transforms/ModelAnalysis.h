#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"

#include "revng/ADT/TypeErasedPtr.hpp"
#include "revng/Model/Binary.h"

namespace mlir::clift {

struct ModelAnalysis {
  ModelAnalysis(mlir::Operation *Op) {}

  TypeErasedPtr<const model::Binary> Binary;
};

inline ModelAnalysis &
getModelAnalysis(mlir::detail::PassExecutionState &State) {
  auto &Analysis = State.analysisManager.getAnalysis<ModelAnalysis>();
  State.preservedAnalyses.preserve<ModelAnalysis>();
  return Analysis;
}

inline void setModel(mlir::detail::PassExecutionState &State,
                     const model::Binary &Model) {
  getModelAnalysis(State)
    .Binary = TypeErasedPtr<const model::Binary>(&Model,
                                                 [](const model::Binary *) {});
}

inline const model::Binary *getModel(mlir::detail::PassExecutionState &State) {
  auto Analysis = State.analysisManager.getCachedAnalysis<ModelAnalysis>();

  if (not Analysis)
    return nullptr;

  State.preservedAnalyses.preserve<ModelAnalysis>();
  return Analysis->get().Binary.get();
}

} // namespace mlir::clift
