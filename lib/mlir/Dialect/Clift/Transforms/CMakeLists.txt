#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_mlir_dialect_library(
  MLIRCliftTransforms
  SHARED
  Beautify/BranchEqualization.cpp
  Beautify/Expressions.cpp
  Beautify/LoopDetection.cpp
  Beautify/Statements.cpp
  Beautify.cpp
  CBackend.cpp
  CSemantics.cpp
  ImportLLVM.cpp
  ModelOption.cpp
  DEPENDS
  LINK_LIBS
  PUBLIC
  MLIRCliftDialect
  MLIRCliftUtils)

revng_register_library(MLIRCliftTransforms revng)
