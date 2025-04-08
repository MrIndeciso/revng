#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/GenericDomTree.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/ScopedExchange.h"
#include "revng/LocalVariables/LocalVariableHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/Support/FunctionTags.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Utils/Identifier.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportLLVM.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"

namespace clift = mlir::clift;
using namespace clift;

namespace {

[[nodiscard]] mlir::OpBuilder::InsertPoint
saveInsertionPointAfter(const mlir::OpBuilder &Builder) {
  mlir::Block *Block = Builder.getInsertionBlock();
  mlir::Block::iterator Point = Builder.getInsertionPoint();

  // End iterator specifies insertion at the start of Block. Non-end iterator
  // specifies insertion *after* the operation referred to by the iterator.
  Point = Point == Block->begin() ? Block->end() : std::prev(Point);

  return mlir::OpBuilder::InsertPoint(Block, Point);
}

void restoreInsertionPointAfter(mlir::OpBuilder &Builder,
                                mlir::OpBuilder::InsertPoint InsertPoint) {
  revng_assert(InsertPoint.isSet());

  mlir::Block *Block = InsertPoint.getBlock();
  mlir::Block::iterator Point = InsertPoint.getPoint();

  // Convert an end iterator back into begin and advance non-end iterators. This
  // is because the builder inserts operations *before* the specified iterator.
  Point = Point == Block->end() ? Block->begin() : std::next(Point);

  Builder.setInsertionPoint(Block, Point);
}

template<typename OpT, typename... ArgsT>
static mlir::OwningOpRef<OpT> createOperation(mlir::MLIRContext *Context,
                                              mlir::Location Location,
                                              ArgsT &&...Args) {
  OpT Op = mlir::OpBuilder(Context).create<OpT>(Location,
                                                std::forward<ArgsT>(Args)...);
  return mlir::OwningOpRef<OpT>(Op);
}

using ScopeGraphPostDomTree = llvm::PostDomTreeOnView<llvm::BasicBlock, Scope>;

class LLVMCodeImporter {
public:
  static mlir::OwningOpRef<clift::ModuleOp> import(mlir::MLIRContext *Context,
                                                   const model::Binary &Model,
                                                   const llvm::Module *Module) {
    return LLVMCodeImporter(Context, Model).importModule(Module);
  }

private:
  explicit LLVMCodeImporter(mlir::MLIRContext *Context,
                            const model::Binary &Model) :
    Context(Context),
    Model(Model),
    Builder(Context),
    PointerSize(getPointerSize(Model.Architecture())) {}

  /* Debug info */

  mlir::Location getLocation(const llvm::DISubprogram *Subprogram) {
    if (Subprogram == nullptr)
      return mlir::UnknownLoc::get(Context);

    auto Content = mlir::StringAttr::get(Context, Subprogram->getName());
    return mlir::NameLoc::get(Content, mlir::UnknownLoc::get(Context));
  }

  mlir::Location getLocation(const llvm::DebugLoc &DL) {
    if (not DL)
      return mlir::UnknownLoc::get(Context);

    const llvm::MDNode *Scope = DL->getScope();
    return getLocation(llvm::dyn_cast_or_null<llvm::DISubprogram>(Scope));
  }

  mlir::Location getLocation(const llvm::BasicBlock *BB) {
    return getLocation(&BB->front());
  }

  mlir::Location getLocation(const llvm::Instruction *I) {
    return getLocation(I->getDebugLoc());
  }

  /* Type utilities */

  mlir::BoolAttr getBoolAttr(bool Value) {
    return mlir::BoolAttr::get(Context, Value);
  }

  static uint64_t getIntegerSize(unsigned IntegerWidth) {
    // Compute the smallest power-of-two number of bytes capable of representing
    // the type based on its bit width:
    return std::bit_ceil((IntegerWidth + 7) / 8);
  }

  clift::ValueType makePointerType(clift::ValueType ElementType) {
    return PointerType::get(ElementType, PointerSize);
  }

  clift::ValueType getVoidType() {
    if (not VoidTypeCache)
      VoidTypeCache = PrimitiveType::get(Context, PrimitiveKind::VoidKind, 0);
    return VoidTypeCache;
  }

  clift::ValueType getVoidPointerType() {
    if (not PointerTypeCache)
      PointerTypeCache = makePointerType(getVoidType());
    return PointerTypeCache;
  }

  clift::ValueType getIntptrType() {
    return PrimitiveType::get(Context, PrimitiveKind::GenericKind, PointerSize);
  }

  clift::ValueType
  getPrimitiveType(uint64_t Size,
                   PrimitiveKind Kind = PrimitiveKind::GenericKind) {
    return PrimitiveType::get(Context, Kind, Size);
  }

  /* Model type import */

  clift::ValueType importModelType(const model::Type &Type) {
    auto EmitError = [&]() -> mlir::InFlightDiagnostic {
      return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                           mlir::DiagnosticSeverity::Error);
    };
    return clift::importModelType(EmitError, *Context, Type, Model);
  }

  clift::ValueType importModelType(const model::TypeDefinition &Type) {
    auto EmitError = [&]() -> mlir::InFlightDiagnostic {
      return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                           mlir::DiagnosticSeverity::Error);
    };
    return clift::importModelType(EmitError, *Context, Type, Model);
  }

  template<typename TypeT, typename ModelTypeT>
  TypeT importModelType(const ModelTypeT &Type) {
    return mlir::cast<TypeT>(importModelType(Type));
  }

  /* LLVM type import */

  clift::ValueType
  importLLVMIntegerType(const llvm::IntegerType *Type,
                        PrimitiveKind Kind = PrimitiveKind::GenericKind) {
    return getPrimitiveType(getIntegerSize(Type->getBitWidth()), Kind);
  }

  clift::ValueType importLLVMPointerType(const llvm::PointerType *Type) {
    revng_assert(Type->getAddressSpace() == 0);
    return getVoidPointerType();
  }

  class LLVMTypeImporter {
  public:
    static clift::ValueType import(LLVMCodeImporter &CodeImporter,
                                   const llvm::Twine &Namespace,
                                   const llvm::Type *Type,
                                   bool MayCreateTypeDefinitions) {
      return LLVMTypeImporter(CodeImporter, Namespace, MayCreateTypeDefinitions)
        .importType(Type);
    }

  private:
    explicit LLVMTypeImporter(LLVMCodeImporter &CodeImporter,
                              const llvm::Twine &Namespace,
                              bool MayCreateTypeDefinitions) :
      CodeImporter(CodeImporter),
      RootHandle(Namespace),
      Handle(&RootHandle),
      MayCreateTypeDefinitions(MayCreateTypeDefinitions) {}

    std::string getHandle() const {
      revng_assert(MayCreateTypeDefinitions);
      return Handle->str();
    }

    RecursiveCoroutine<clift::ValueType>
    importArrayType(const llvm::ArrayType *Type) {
      auto ElementType = rc_recur importType(Type->getElementType());
      rc_return ArrayType::get(ElementType, Type->getNumElements());
    }

    class ScopedHandle {
    public:
      explicit ScopedHandle(LLVMTypeImporter &Importer,
                            auto ConstructLeafTwine) :
        LeafTwine(ConstructLeafTwine()),
        BranchTwine(*Importer.Handle + LeafTwine),
        Exchange(Importer.Handle, &BranchTwine) {}

    private:
      llvm::Twine LeafTwine;
      llvm::Twine BranchTwine;

      ScopedExchange<const llvm::Twine *> Exchange;
    };

    ScopedHandle appendHandle(const auto &Value) {
      return ScopedHandle(*this,
                          [&]() -> llvm::Twine { return llvm::Twine(Value); });
    }

    RecursiveCoroutine<clift::ValueType>
    importFunctionType(const llvm::FunctionType *Type) {
      revng_assert(not Type->isVarArg());

      clift::ValueType ReturnType = (appendHandle("/return-type"),
                                     importType(Type->getReturnType()));

      llvm::SmallVector<clift::ValueType> ParameterTypes;
      ParameterTypes.reserve(Type->getNumParams());

      for (auto [I, T] : llvm::enumerate(Type->params()))
        appendHandle("/parameter-type/"), appendHandle(I),
          ParameterTypes.push_back(rc_recur importType(T));

      rc_return FunctionType::get(CodeImporter.Context,
                                  getHandle(),
                                  "",
                                  ReturnType,
                                  ParameterTypes);
    }

    RecursiveCoroutine<clift::ValueType>
    importStructType(const llvm::StructType *Type) {
      revng_assert(Type->isLiteral());

      llvm::SmallVector<FieldAttr> Fields;
      Fields.reserve(Type->getNumElements());

      uint64_t Offset = 0;
      for (const llvm::Type *T : Type->elements()) {
        clift::ValueType FieldType = rc_recur importType(T);

        Fields.push_back(FieldAttr::get(CodeImporter.Context,
                                        Offset,
                                        FieldType,
                                        ""));

        Offset += FieldType.getByteSize();
      }

      rc_return StructType::get(CodeImporter.Context,
                                getHandle(),
                                "",
                                Offset,
                                Fields);
    }

    RecursiveCoroutine<clift::ValueType> importType(const llvm::Type *Type) {
      if (Type->isVoidTy())
        rc_return CodeImporter.getVoidType();

      if (auto *T = llvm::dyn_cast<llvm::IntegerType>(Type))
        rc_return CodeImporter.importLLVMIntegerType(T);

      if (auto *T = llvm::dyn_cast<llvm::PointerType>(Type))
        rc_return CodeImporter.importLLVMPointerType(T);

      if (auto *T = llvm::dyn_cast<llvm::ArrayType>(Type))
        rc_return rc_recur importArrayType(T);

      if (auto *T = llvm::dyn_cast<llvm::FunctionType>(Type))
        rc_return rc_recur importFunctionType(T);

      if (auto *T = llvm::dyn_cast<llvm::StructType>(Type))
        rc_return rc_recur importStructType(T);

      revng_abort("Unsupported LLVM type");
    }

    LLVMCodeImporter &CodeImporter;
    llvm::Twine RootHandle;
    const llvm::Twine *Handle;
    bool MayCreateTypeDefinitions;
  };

  clift::ValueType importLLVMType(const llvm::Type *Type) {
    llvm::Twine Namespace = {};
    return LLVMTypeImporter::import(*this, Namespace, Type, false);
  }

  clift::ValueType importLLVMType(const llvm::Type *Type,
                                  const llvm::Twine &Namespace) {
    return LLVMTypeImporter::import(*this, Namespace, Type, true);
  }

  /* LLVM expression import */

  uint64_t getConstantInt(const llvm::Value *Value) {
    return llvm::cast<llvm::ConstantInt>(Value)->getZExtValue();
  }

  clift::FunctionType importHelperType(const llvm::Function *F) {
    llvm::Twine HandleRoot = llvm::Twine("/helper/");
    auto T = importLLVMType(F->getFunctionType(),
                            HandleRoot + llvm::Twine(F->getName()));
    return mlir::cast<clift::FunctionType>(T);
  }

  clift::FunctionOp emitFunctionDeclaration(const llvm::Function *F) {
    const model::Function *MF = llvmToModelFunction(Model, *F);

    auto
      FunctionType = MF != nullptr ?
                       importModelType<clift::FunctionType>(*MF->Prototype()) :
                       importHelperType(F);

    return Builder.create<FunctionOp>(getLocation(F->getSubprogram()),
                                      F->getName(),
                                      FunctionType);
  }

  clift::GlobalVariableOp
  emitVariableDeclaration(const llvm::GlobalVariable *V) {
    clift::ValueType VariableType = importLLVMType(V->getType());

    return Builder.create<GlobalVariableOp>(mlir::UnknownLoc::get(Context),
                                            V->getName(),
                                            VariableType);
  }

  clift::ValueType emitGlobalObject(const llvm::GlobalObject *G) {
    auto [Iterator, Inserted] = SymbolMapping.try_emplace(G);
    if (Inserted) {
      Iterator->second = [&]() -> clift::GlobalOpInterface {
        mlir::OpBuilder::InsertionGuard Guard(Builder);
        Builder.setInsertionPointToEnd(&CurrentModule.getBody().front());

        if (auto *F = llvm::dyn_cast<llvm::Function>(G))
          return emitFunctionDeclaration(F);

        if (auto *V = llvm::dyn_cast<llvm::GlobalVariable>(G))
          return emitVariableDeclaration(V);

        revng_abort("Unsupported global object kind");
      }();
    }
    return Iterator->second.getType();
  }

  template<typename OpT, typename... ArgsT>
  mlir::Value emitExpr(mlir::Location Loc, ArgsT &&...Args) {
    return Builder.create<OpT>(Loc, std::forward<ArgsT>(Args)...);
  }

  mlir::Value emitCast(mlir::Location Loc,
                       mlir::Value Value,
                       clift::ValueType TargetType,
                       CastKind Kind = CastKind::Bitcast) {
    if (Value.getType() != TargetType)
      Value = Builder.create<CastOp>(Loc, TargetType, Value, Kind);

    return Value;
  }

  mlir::Value emitImplicitCast(mlir::Location Loc,
                               mlir::Value Value,
                               clift::ValueType TargetType) {
    auto SourceType = mlir::cast<clift::ValueType>(Value.getType());

    if (SourceType == TargetType)
      return Value;

    auto UnderlyingSourceT = dealias(SourceType, true);
    auto UnderlyingTargetT = dealias(TargetType, true);

    if (UnderlyingSourceT.getByteSize() != UnderlyingTargetT.getByteSize())
      return Value;

    if (mlir::isa<ArrayType>(UnderlyingSourceT))
      return Value;

    if (mlir::isa<ArrayType>(UnderlyingTargetT))
      return Value;

    return emitCast(Loc, Value, TargetType);
  }

  mlir::Value emitIntegerOp(mlir::Location Loc,
                            PrimitiveKind Kind,
                            auto ApplyOperation,
                            std::same_as<mlir::Value> auto... Operands) {
    auto ConvertToKind = [&](mlir::Value &Value, PrimitiveKind Kind) {
      uint64_t Size = getUnderlyingIntegerType(Value.getType()).getSize();
      Value = emitCast(Loc, Value, getPrimitiveType(Size, Kind));
    };

    (ConvertToKind(Operands, Kind), ...);
    mlir::Value Result = ApplyOperation(Operands...);

    if (Kind != PrimitiveKind::GenericKind)
      ConvertToKind(Result, PrimitiveKind::GenericKind);

    return Result;
  }

  mlir::Value emitIntegerCast(mlir::Location Loc,
                              mlir::Value Operand,
                              uint64_t Size,
                              PrimitiveKind Kind) {
    uint64_t SrcSize = getUnderlyingIntegerType(Operand.getType()).getSize();

    CastKind Cast = CastKind::Bitcast;
    if (Size > SrcSize)
      Cast = CastKind::Extend;
    if (Size < SrcSize)
      Cast = CastKind::Truncate;

    auto EmitCast = [&](mlir::Value Operand) {
      return emitCast(Loc, Operand, getPrimitiveType(Size, Kind), Cast);
    };

    return emitIntegerOp(Loc, Kind, EmitCast, Operand);
  }

  std::string makeLocalVariableName() {
    std::string Name;
    {
      llvm::raw_string_ostream Out(Name);
      Out << "_local_" << LocalCount++;
    }
    return Name;
  }

  clift::FunctionOp emitHelperDeclaration(clift::FunctionType FunctionType) {
    std::string Name = clift::sanitizeIdentifier(FunctionType.getHandle());

    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToEnd(&CurrentModule.getBody().front());

    return Builder.create<FunctionOp>(mlir::UnknownLoc::get(Context),
                                      Name,
                                      FunctionType);
  }

  mlir::Value getHelperFunction(mlir::Location SurroundingLocation,
                                clift::FunctionType FunctionType) {
    auto [Iterator, Inserted] = HelperFunctionMapping
                                  .try_emplace(FunctionType.getHandle());

    if (Inserted)
      Iterator->second = emitHelperDeclaration(FunctionType);

    return Builder.create<UseOp>(SurroundingLocation,
                                 FunctionType,
                                 Iterator->second.getName());
  }

  mlir::Value emitHelperCall(mlir::Location Loc,
                             clift::FunctionType FunctionType,
                             llvm::ArrayRef<mlir::Value> Arguments) {
    mlir::Value Function = getHelperFunction(Loc, FunctionType);

    llvm::SmallVector<mlir::Value> CastArgs;
    CastArgs.reserve(Arguments.size());

    for (auto [A, T] : llvm::zip(Arguments, FunctionType.getArgumentTypes()))
      CastArgs.push_back(emitImplicitCast(Loc, A, T));

    return Builder.create<CallOp>(Loc,
                                  FunctionType.getReturnType(),
                                  Function,
                                  CastArgs);
  }

  mlir::Value emitHelperCall(mlir::Location Loc,
                             llvm::StringRef Handle,
                             mlir::Type ReturnType,
                             llvm::ArrayRef<mlir::Type> ParameterTypes,
                             llvm::ArrayRef<mlir::Value> Arguments) {
    auto FunctionType = clift::FunctionType::get(Context,
                                                 Handle,
                                                 "",
                                                 ReturnType,
                                                 ParameterTypes);

    return emitHelperCall(Loc, FunctionType, Arguments);
  }

  RecursiveCoroutine<mlir::Value>
  emitOpaqueExtractValue(const llvm::CallInst *Call) {
    mlir::Location Loc = getLocation(Call);

    mlir::Value Aggregate = rc_recur emitExpression(Call->getArgOperand(0),
                                                    Loc);

    uint64_t Index = getConstantInt(Call->getArgOperand(1));
    auto Struct = mlir::cast<StructType>(Aggregate.getType());
    //llvm::errs() << *Call << "\n";
    revng_assert(Index < Struct.getFields().size());
    mlir::Type ResultType = Struct.getFields()[Index].getType();

    rc_return Builder.create<AccessOp>(Loc,
                                       ResultType,
                                       Aggregate,
                                       /*indirect=*/false,
                                       Index);
  }

  RecursiveCoroutine<mlir::Value> emitHelperCall(const llvm::CallInst *Call) {
    const llvm::Function *Function = Call->getCalledFunction();
    auto Tags = FunctionTags::TagsSet::from(Function);

    if (Tags.contains(FunctionTags::OpaqueExtractValue))
      rc_return rc_recur emitOpaqueExtractValue(Call);

    mlir::Location Loc = getLocation(Call);

    llvm::SmallVector<mlir::Value> Arguments;
    Arguments.reserve(Call->arg_size());

    for (const llvm::Value *Argument : Call->args())
      Arguments.push_back(rc_recur emitExpression(Argument, Loc));

    rc_return emitHelperCall(Loc, importHelperType(Function), Arguments);
  }

  struct StringLiteral {
    clift::ArrayType Type;
    std::string Data;
  };

  std::optional<StringLiteral>
  deduceStringLiteral(const llvm::GlobalObject *O) {
    const auto *V = llvm::dyn_cast<llvm::GlobalVariable>(O);
    if (V == nullptr)
      return std::nullopt;

    if (not V->isConstant())
      return std::nullopt;

    const auto *Type = llvm::dyn_cast<llvm::ArrayType>(V->getValueType());
    if (Type == nullptr)
      return std::nullopt;

    const auto
      *ElementType = llvm::dyn_cast<llvm::IntegerType>(Type->getElementType());

    if (ElementType == nullptr or ElementType->getBitWidth() != 8)
      return std::nullopt;

    const llvm::Constant *Initializer = V->getInitializer();
    if (Initializer == nullptr)
      return std::nullopt;
    revng_assert(Initializer->getType() == Type);

    const auto *Array = llvm::dyn_cast<llvm::ConstantDataArray>(Initializer);
    if (Array == nullptr)
      return std::nullopt;

    revng_assert(Type->getNumElements() != 0);
    unsigned Length = Type->getNumElements() - 1;

    const auto GetChar = [&](unsigned Index) -> uint8_t {
      uint64_t Value = getConstantInt(Array->getAggregateElement(Index));
      revng_assert(Value < 0x100);
      return Value;
    };

    if (GetChar(Length) != 0)
      return std::nullopt;

    std::optional<StringLiteral> Result(std::in_place);

    auto CharType = clift::PrimitiveType::get(Context,
                                              PrimitiveKind::NumberKind,
                                              1,
                                              /*IsConst=*/true);

    Result->Type = clift::ArrayType::get(CharType, Type->getNumElements());

    Result->Data.reserve(Length);
    for (unsigned I = 0; I < Length; ++I)
      Result->Data.push_back(GetChar(I));

    return Result;
  }

  RecursiveCoroutine<mlir::Value>
  emitExpression(const llvm::Value *V, mlir::Location SurroundingLocation) {
    if (auto G = llvm::dyn_cast<llvm::GlobalObject>(V)) {
      if (auto OptString = deduceStringLiteral(G)) {
        auto Op = Builder.create<StringOp>(SurroundingLocation,
                                           OptString->Type,
                                           std::move(OptString->Data));

        auto Type = makePointerType(OptString->Type.getElementType());

        rc_return emitCast(SurroundingLocation, Op, Type, CastKind::Decay);
      }

      rc_return Builder.create<UseOp>(SurroundingLocation,
                                      emitGlobalObject(G),
                                      G->getName());
    }

    if (auto It = ValueMapping.find(V); It != ValueMapping.end())
      rc_return It->second;

    if (auto A = llvm::dyn_cast<llvm::Argument>(V)) {
      auto It = ArgumentMapping.find(A);
      revng_assert(It != ArgumentMapping.end());

      mlir::Value Value = It->second.Argument;

      if (mlir::Type CastType = It->second.CastType)
        Value = emitImplicitCast(SurroundingLocation, Value, CastType);

      rc_return Value;
    }

    if (auto U = llvm::dyn_cast<llvm::UndefValue>(V)) {
      mlir::Type Type = importLLVMType(U->getType());
      rc_return Builder.create<UndefOp>(SurroundingLocation, Type);
    }

    if (auto C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      const llvm::IntegerType *T = llvm::cast<llvm::IntegerType>(C->getType());
      rc_return Builder.create<ImmediateOp>(SurroundingLocation,
                                            importLLVMIntegerType(T),
                                            C->getZExtValue());
    }

    if (auto N = llvm::dyn_cast<llvm::ConstantPointerNull>(V)) {
      auto Op = Builder.create<ImmediateOp>(SurroundingLocation,
                                            getIntptrType(),
                                            /*Value=*/0);

      rc_return emitCast(SurroundingLocation, Op, getVoidPointerType());
    }

    if (auto E = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
      // TODO: This could be made more efficient by implementing direct
      //       ConstantExpr import.
      llvm::Instruction *I = E->getAsInstruction();
      mlir::Value Value = emitExpression(I, SurroundingLocation);
      I->deleteValue();
      rc_return Value;
    }

    if (auto I = llvm::dyn_cast<llvm::AllocaInst>(V)) {
      mlir::Location Loc = getLocation(I);

      if (auto It = AllocaMapping.find(I); It != AllocaMapping.end()) {
        auto Type = makePointerType(It->second.getType());
        rc_return Builder.create<AddressofOp>(Loc, Type, It->second);
      }

      clift::ValueType Type = importLLVMType(I->getAllocatedType());
      Type = makePointerType(Type);

      revng_assert(not llvm::isa<llvm::Constant>(I->getArraySize()));
      revng_abort("Non-constant alloca is not supported.");
    }

    if (auto I = llvm::dyn_cast<llvm::LoadInst>(V)) {
      mlir::Location Loc = getLocation(I);

      mlir::Value Pointer = rc_recur emitExpression(I->getPointerOperand(),
                                                    Loc);

      clift::ValueType ValueType = importLLVMType(V->getType());
      clift::ValueType PointerType = makePointerType(ValueType);

      auto Op1 = Builder.create<CastOp>(Loc,
                                        PointerType,
                                        Pointer,
                                        CastKind::Bitcast);

      rc_return Builder.create<IndirectionOp>(Loc, ValueType, Op1);
    }

    if (auto I = llvm::dyn_cast<llvm::StoreInst>(V)) {
      mlir::Location Loc = getLocation(I);

      mlir::Value Pointer = rc_recur emitExpression(I->getPointerOperand(),
                                                    Loc);
      mlir::Value Value = rc_recur emitExpression(I->getValueOperand(), Loc);

      auto Op1 = Builder.create<CastOp>(Loc,
                                        makePointerType(Value.getType()),
                                        Pointer,
                                        CastKind::Bitcast);

      auto Op2 = Builder.create<IndirectionOp>(Loc, Value.getType(), Op1);

      rc_return Builder.create<AssignOp>(Loc, Value.getType(), Op2, Value);
    }

    if (auto I = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
      using Operators = llvm::BinaryOperator::BinaryOps;

      mlir::Location Loc = getLocation(I);

      mlir::Value Lhs = rc_recur emitExpression(I->getOperand(0), Loc);
      mlir::Value Rhs = rc_recur emitExpression(I->getOperand(1), Loc);

      auto LhsPointerType = mlir::dyn_cast<PointerType>(Lhs.getType());
      auto RhsPointerType = mlir::dyn_cast<PointerType>(Rhs.getType());

      if (LhsPointerType or RhsPointerType) {
        switch (I->getOpcode()) {
        case Operators::Add: {
          revng_assert(not LhsPointerType or not RhsPointerType);
          auto Type = LhsPointerType ? LhsPointerType : RhsPointerType;
          rc_return emitExpr<PtrAddOp>(Loc, Type, Lhs, Rhs);
        }

        case Operators::Sub:
          if (LhsPointerType and RhsPointerType) {
            revng_assert(LhsPointerType == RhsPointerType);
            auto Type = getPrimitiveType(LhsPointerType.getPointerSize(),
                                         PrimitiveKind::SignedKind);
            rc_return emitExpr<PtrDiffOp>(Loc, Type, Lhs, Rhs);
          } else {
            auto Type = LhsPointerType ? LhsPointerType : RhsPointerType;
            rc_return emitExpr<PtrSubOp>(Loc, Type, Lhs, Rhs);
          }

        default:
          revng_abort("Unsupported pointer arithmetic operation.");
        }
      }

      PrimitiveKind Kind = PrimitiveKind::GenericKind;
      switch (I->getOpcode()) {
      case Operators::SDiv:
      case Operators::SRem:
      case Operators::AShr:
        Kind = PrimitiveKind::SignedKind;
        break;
      case Operators::UDiv:
      case Operators::URem:
      case Operators::LShr:
        Kind = PrimitiveKind::UnsignedKind;
        break;
      default:
        break;
      }

      auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
      auto Type = importLLVMIntegerType(IntegerType, Kind);

      auto EmitOp = [&](mlir::Value Lhs, mlir::Value Rhs) {
        switch (I->getOpcode()) {
        case Operators::Add:
          return emitExpr<AddOp>(Loc, Type, Lhs, Rhs);
        case Operators::Sub:
          return emitExpr<SubOp>(Loc, Type, Lhs, Rhs);
        case Operators::Mul:
          return emitExpr<MulOp>(Loc, Type, Lhs, Rhs);
        case Operators::SDiv:
        case Operators::UDiv:
          return emitExpr<DivOp>(Loc, Type, Lhs, Rhs);
        case Operators::SRem:
        case Operators::URem:
          return emitExpr<RemOp>(Loc, Type, Lhs, Rhs);
        case Operators::Shl:
          return emitExpr<ShiftLeftOp>(Loc, Type, Lhs, Rhs);
        case Operators::LShr:
        case Operators::AShr:
          return emitExpr<ShiftRightOp>(Loc, Type, Lhs, Rhs);
        case Operators::And:
          return emitExpr<BitwiseAndOp>(Loc, Type, Lhs, Rhs);
        case Operators::Or:
          return emitExpr<BitwiseOrOp>(Loc, Type, Lhs, Rhs);
        case Operators::Xor:
          return emitExpr<BitwiseXorOp>(Loc, Type, Lhs, Rhs);
        default:
          revng_abort("Unsupported LLVM binary operator.");
        }
      };

      rc_return emitIntegerOp(Loc, Kind, EmitOp, Lhs, Rhs);
    }

    if (auto I = llvm::dyn_cast<llvm::ICmpInst>(V)) {
      using enum llvm::ICmpInst::Predicate;

      PrimitiveKind Kind = PrimitiveKind::GenericKind;
      switch (I->getPredicate()) {
      case ICMP_SGT:
      case ICMP_SGE:
      case ICMP_SLT:
      case ICMP_SLE:
        Kind = PrimitiveKind::SignedKind;
        break;
      case ICMP_UGT:
      case ICMP_UGE:
      case ICMP_ULT:
      case ICMP_ULE:
        Kind = PrimitiveKind::UnsignedKind;
        break;
      default:
        break;
      }

      mlir::Location Loc = getLocation(I);
      mlir::Value Lhs = rc_recur emitExpression(I->getOperand(0), Loc);
      mlir::Value Rhs = rc_recur emitExpression(I->getOperand(1), Loc);

      auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
      auto Type = importLLVMIntegerType(IntegerType, PrimitiveKind::SignedKind);

      auto EmitOp = [&](mlir::Value Lhs, mlir::Value Rhs) {
        switch (I->getPredicate()) {
        case ICMP_EQ:
          return emitExpr<CmpEqOp>(Loc, Type, Lhs, Rhs);
        case ICMP_NE:
          return emitExpr<CmpNeOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SGT:
        case ICMP_UGT:
          return emitExpr<CmpGtOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SGE:
        case ICMP_UGE:
          return emitExpr<CmpGeOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SLT:
        case ICMP_ULT:
          return emitExpr<CmpLtOp>(Loc, Type, Lhs, Rhs);
        case ICMP_SLE:
        case ICMP_ULE:
          return emitExpr<CmpLeOp>(Loc, Type, Lhs, Rhs);
        default:
          revng_abort("Unsupported LLVM comparison predicate.");
        }
      };

      rc_return emitIntegerOp(Loc, Kind, EmitOp, Lhs, Rhs);
    }

    if (auto I = llvm::dyn_cast<llvm::CastInst>(V)) {
      mlir::Location Loc = getLocation(I);
      mlir::Value Operand = rc_recur emitExpression(I->getOperand(0), Loc);

      auto emitIntegerCast = [&](PrimitiveKind Kind) {
        auto *IntegerType = llvm::cast<llvm::IntegerType>(V->getType());
        uint64_t TargetSize = getIntegerSize(IntegerType->getBitWidth());
        return this->emitIntegerCast(Loc, Operand, TargetSize, Kind);
      };

      switch (I->getOpcode()) {
        using Operators = llvm::CastInst::CastOps;
      case Operators::Trunc:
        rc_return emitIntegerCast(PrimitiveKind::GenericKind);
      case Operators::SExt:
        rc_return emitIntegerCast(PrimitiveKind::SignedKind);
      case Operators::ZExt:
        rc_return emitIntegerCast(PrimitiveKind::UnsignedKind);
      case Operators::PtrToInt:
        rc_return emitCast(Loc, Operand, getIntptrType());
      case Operators::IntToPtr:
        rc_return emitCast(Loc, Operand, getVoidPointerType());
      default:
        revng_abort("Unsupported LLVM cast operation.");
      }
    }

    if (auto I = llvm::dyn_cast<llvm::CallInst>(V)) {
      mlir::Location Loc = getLocation(I);

      if (not I->hasMetadata(PrototypeMDName))
        rc_return emitHelperCall(I);

      const auto *ModelCallType = getCallSitePrototype(Model, I);
      auto Layout = abi::FunctionType::Layout::make(*ModelCallType);
      //llvm::errs() << *I << "\n";
      //llvm::errs() << "  RM=" << (int)Layout.returnMethod() << "\n";
      //llvm::errs() << "  SPTAR=" << (int)Layout.hasSPTAR() << "\n";

      auto CallType = importModelType<clift::FunctionType>(*ModelCallType);

      mlir::Value Function = rc_recur emitExpression(I->getCalledOperand(),
                                                     Loc);

      clift::FunctionType
        FunctionType = getFunctionOrFunctionPointerFunctionType(Function
                                                                  .getType());

      if (CallType != FunctionType) {
        if (mlir::isa<clift::FunctionType>(Function.getType())) {
          Function = emitCast(Loc,
                              Function,
                              makePointerType(FunctionType),
                              CastKind::Decay);
        }

        Function = emitCast(Loc, Function, makePointerType(CallType));
      }

      auto ReturnType = CallType.getReturnType();

      llvm::SmallVector<mlir::Value> Arguments;
      for (auto [A, T] : llvm::zip(I->args(), CallType.getArgumentTypes()))
        Arguments.push_back(emitImplicitCast(Loc,
                                             rc_recur emitExpression(A, Loc),
                                             T));

      rc_return Builder.create<CallOp>(Loc, ReturnType, Function, Arguments);
    }

    if (auto I = llvm::dyn_cast<llvm::SelectInst>(V)) {
      mlir::Location Loc = getLocation(I);

      mlir::Value Condition = rc_recur emitExpression(I->getCondition(), Loc);
      mlir::Value True = rc_recur emitExpression(I->getTrueValue(), Loc);
      mlir::Value False = rc_recur emitExpression(I->getFalseValue(), Loc);

      auto TrueType = mlir::cast<clift::ValueType>(True.getType());
      auto FalseType = mlir::cast<clift::ValueType>(False.getType());

      auto ResultType = TrueType.removeConst();
      if (ResultType != FalseType.removeConst()) {
        ResultType = importLLVMType(I->getType());
        True = emitImplicitCast(Loc, True, ResultType);
        False = emitImplicitCast(Loc, False, ResultType);
      }

      rc_return Builder.create<TernaryOp>(Loc,
                                          ResultType,
                                          Condition,
                                          True,
                                          False);
    }

    if (auto I = llvm::dyn_cast<llvm::GetElementPtrInst>(V)) {
      auto Alloca = llvm::cast<llvm::AllocaInst>(I->getPointerOperand());

      auto It = AllocaMapping.find(Alloca);
      revng_assert(It != AllocaMapping.end());

      auto AT = mlir::cast<clift::ArrayType>(It->second.getType());
      auto PT = makePointerType(AT.getElementType());

      revng_assert(I->getNumIndices() == 2);
      auto IndexIterator = I->idx_begin();

      revng_assert(getConstantInt(IndexIterator->get()) == 0);
      uint64_t Index1 = getConstantInt((++IndexIterator)->get());

      mlir::Location Loc = getLocation(I);
      auto Operand = emitCast(Loc, It->second, PT, CastKind::Decay);

      mlir::Value Immediate = emitExpr<ImmediateOp>(Loc,
                                                    getIntptrType(),
                                                    Index1);

      rc_return emitExpr<PtrAddOp>(Loc, PT, Operand, Immediate);
    }

    if (auto I = llvm::dyn_cast<llvm::FreezeInst>(V))
      rc_return emitExpression(I->getOperand(0), getLocation(I));

    revng_abort("Unsupported LLVM instruction.");
  }

  mlir::Type emitExpressionTreeImpl(mlir::Block &B, auto EmitExpression) {
    revng_assert(B.empty());

    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPointToEnd(&B);

    mlir::Value Value = EmitExpression();
    Builder.create<YieldOp>(Value.getLoc(), Value);
    return Value.getType();
  }

  mlir::Type emitExpressionTreeImpl(mlir::Region &R, auto EmitExpression) {
    revng_assert(R.empty());
    return emitExpressionTreeImpl(R.emplaceBlock(), EmitExpression);
  }

  mlir::Type emitExpressionTree(mlir::Block &B,
                                const llvm::Value *V,
                                mlir::Location SurroundingLocation) {
    return emitExpressionTreeImpl(B, [&]() {
      return emitExpression(V, SurroundingLocation);
    });
  }

  mlir::Type emitExpressionTree(mlir::Region &R,
                                const llvm::Value *V,
                                mlir::Location SurroundingLocation) {
    return emitExpressionTreeImpl(R, [&]() {
      return emitExpression(V, SurroundingLocation);
    });
  }

  [[nodiscard]] std::pair<mlir::Block *, mlir::Type>
  emitExpressionTree(const llvm::Value *V, mlir::Location SurroundingLocation) {
    mlir::Block *B = new mlir::Block();
    mlir::Type Type = emitExpressionTree(*B, V, SurroundingLocation);
    return { B, Type };
  }

  /* LLVM control flow import */

  static mlir::Block::iterator getLabelInsertionPoint(mlir::Block *Block) {
    mlir::Block::iterator Beg = Block->begin();
    mlir::Block::iterator End = Block->end();

    while (Beg != End and mlir::isa<MakeLabelOp>(*Beg))
      ++Beg;

    return Beg;
  }

  mlir::Value emitMakeLabel(mlir::Location Loc) {
    mlir::OpBuilder::InsertionGuard Guard(Builder);
    Builder.setInsertionPoint(LabelBlock, getLabelInsertionPoint(LabelBlock));

    std::string Name;
    {
      llvm::raw_string_ostream Out(Name);
      Out << "BB_" << LabelCount++;
    }

    return Builder.create<MakeLabelOp>(Loc, Name);
  }

  void emitAssignLabel(mlir::Value Label, mlir::Location Loc) {
    Builder.create<AssignLabelOp>(Loc, Label);
  }

  static bool isScopeGraphEdge(const llvm::BasicBlock *Pred,
                               const llvm::BasicBlock *Succ) {
    for (auto *BB : llvm::children<Scope<const llvm::BasicBlock *>>(Pred)) {
      if (BB == Succ)
        return true;
    }
    return false;
  }

  static bool isUsedOutsideOfBlock(const llvm::Value *V,
                                   const llvm::BasicBlock *BB) {
    for (const llvm::User *U : V->users()) {
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (I->getParent() != BB)
          return false;
      }
    }
    return true;
  }

  bool hasBeenEmitted(const llvm::BasicBlock *BB) {
    auto It = BlockMapping.find(BB);
    return It != BlockMapping.end() and It->second.InsertPoint.isSet();
  }

  void emitGoto(mlir::Location Loc, const llvm::BasicBlock *BB) {
    auto [Iterator, Inserted] = BlockMapping.try_emplace(BB);

    if (not Iterator->second.Label)
      Iterator->second.Label = emitMakeLabel(getLocation(BB));

    if (not Iterator->second.HasAssignLabel
        and Iterator->second.InsertPoint.isSet()) {
      mlir::OpBuilder::InsertionGuard Guard(Builder);
      restoreInsertionPointAfter(Builder, Iterator->second.InsertPoint);
      emitAssignLabel(Iterator->second.Label, getLocation(BB));
      Iterator->second.HasAssignLabel = true;
    }

    Builder.create<GoToOp>(Loc, Iterator->second.Label);
  }

  bool requiresFullExpression(const llvm::CallInst *Call) {
    if (Call->hasMetadata(PrototypeMDName)) {
      const auto *ModelCallType = getCallSitePrototype(Model, Call);
      auto Layout = abi::FunctionType::Layout::make(*ModelCallType);
      namespace ReturnMethod = abi::FunctionType::ReturnMethod;
      return Layout.returnMethod() == ReturnMethod::RegisterSet;
    }

    return llvm::isa<llvm::StructType>(Call->getType());
  }

  RecursiveCoroutine<void>
  emitBasicBlock(const llvm::BasicBlock *BB,
                 const llvm::BasicBlock *InnerPostDom,
                 const llvm::BasicBlock *OuterPostDom) {
    // Map BB to the MLIR block, emit label if necessary:
    {
      auto [Iterator, Inserted] = BlockMapping.try_emplace(BB);

      revng_assert(not Iterator->second.InsertPoint.isSet());
      Iterator->second.InsertPoint = saveInsertionPointAfter(Builder);

      if (not Inserted and Iterator->second.Label) {
        emitAssignLabel(Iterator->second.Label, getLocation(BB));
        Iterator->second.HasAssignLabel = true;
      }
    }

    const llvm::Instruction *Terminal = BB->getTerminator();
    bool HasGotoMarker = false;

    for (const llvm::Instruction &I : *BB) {
      if (&I == Terminal)
        break;

      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        const auto *Size = Alloca->getArraySize();

        // Non-constant alloca is not supported:
        revng_assert(not Size or llvm::isa<llvm::ConstantInt>(Size));

        clift::ValueType Type;
        if (hasStackTypeMetadata(Alloca)) {
          Type = importModelType(*getStackTypeFromMetadata(Alloca, Model));
        } else if (hasVariableTypeMetadata(Alloca)) {
          Type = importModelType(*getVariableTypeFromMetadata(Alloca, Model));
        } else {
          Type = importLLVMType(Alloca->getAllocatedType());

          if (Alloca->isArrayAllocation())
            Type = ArrayType::get(Context, Type, getConstantInt(Size));
        }

        auto Op = Builder.create<LocalVariableOp>(getLocation(Alloca),
                                                  Type,
                                                  makeLocalVariableName());

        auto [Iterator, Inserted] = AllocaMapping.try_emplace(Alloca, Op);
        revng_assert(Inserted);

        continue;
      }

      if (const llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (isCallToTagged(Call, FunctionTags::GotoBlockMarker)) {
          HasGotoMarker = true;
          continue;
        }

        // Scope closer markers are just ignored. They only affect the scope
        // graph structure.
        if (isCallToTagged(Call, FunctionTags::ScopeCloserMarker))
          continue;

        // Some function calls are emitted in local variable initializers.
        if (requiresFullExpression(Call)) {
          auto [Block, Type] = emitExpressionTree(Call, getLocation(Call));

          auto Op = Builder.create<LocalVariableOp>(getLocation(Call),
                                                    Type,
                                                    makeLocalVariableName());

          Op.getInitializer().push_back(Block);

          auto [Iterator, Inserted] = ValueMapping.try_emplace(Call, Op);
          revng_assert(Inserted);

          continue;
        }
      }

      if (I.use_empty()) {
        // Any instruction with no uses can be considered the root of an
        // expression tree. The tree can be emitted in an expression statement.
        mlir::Location Loc = getLocation(&I);
        auto Op = Builder.create<ExpressionStatementOp>(Loc);
        emitExpressionTree(Op.getExpression(), &I, Loc);
      }

#if 0
      if (I.use_empty()) {
        // Any instruction with no uses can be considered the root of an
        // expression tree. The tree can be emitted in an expression statement.
        auto Op = Builder.create<ExpressionStatementOp>(mlir::UnknownLoc::get(Context));
        emitExpressionTree(&I, Op.getExpression());
      } else if (I.hasNUsesOrMore(2) or isUsedOutsideOfBlock(&I, BB)) {
        mlir::Region R;
        mlir::Type Type = emitExpressionTree(&I, R);

        // Any instruction with more than one use, or with a single use outside
        // of this block must be emitted into a local variable initializer.
        auto Op = Builder.create<LocalVariableOp>(mlir::UnknownLoc::get(Context),
                                                  Type,
                                                  "");

        // Move the local block into the initializer region.
        Op.getInitializer().push_back(R.getBlocks().remove(R.front()));

        // Map this instruction value to the newly created local variable.
        auto [It, Inserted] = ValueMapping.try_emplace(&I, Op);
        revng_assert(Inserted);
      }
#endif
    }

    if (Terminal->getNumSuccessors() == 1) {
      const llvm::BasicBlock *Succ = Terminal->getSuccessor(0);

      if (HasGotoMarker)
        emitGoto(getLocation(Terminal), Succ);
      else
        rc_recur emitBasicBlock(Succ, InnerPostDom);

      rc_return;
    }

    revng_assert(not HasGotoMarker);
    mlir::Location TerminalLoc = getLocation(Terminal);

    if (llvm::isa<llvm::UnreachableInst>(Terminal)) {
      auto Op = Builder.create<ExpressionStatementOp>(TerminalLoc);

      emitExpressionTreeImpl(Op.getExpression(), [&]() {
        return emitHelperCall(TerminalLoc,
                              "/helper/generic/unreachable",
                              getVoidType(),
                              {},
                              {});
      });
    } else if (auto *Return = llvm::dyn_cast<llvm::ReturnInst>(Terminal)) {
      auto Op = Builder.create<ReturnOp>(TerminalLoc);
      if (const llvm::Value *Value = Return->getReturnValue()) {
        auto FunctionType = CurrentFunction.getCliftFunctionType();
        auto ReturnType = FunctionType.getReturnType();

        emitExpressionTreeImpl(Op.getResult(), [&]() {
          mlir::Value ReturnValue = emitExpression(Value, TerminalLoc);

          if (auto AT = mlir::dyn_cast<ArrayType>(ReturnValue.getType())) {
            ReturnValue = emitCast(TerminalLoc,
                                   ReturnValue,
                                   makePointerType(AT.getElementType()),
                                   CastKind::Decay);

            ReturnValue = emitCast(TerminalLoc,
                                   ReturnValue,
                                   makePointerType(ReturnType),
                                   CastKind::Bitcast);

            ReturnValue = Builder.create<IndirectionOp>(TerminalLoc,
                                                        ReturnType,
                                                        ReturnValue);
          }

          return emitImplicitCast(TerminalLoc, ReturnValue, ReturnType);
        });
      }
    } else if (auto *Branch = llvm::dyn_cast<llvm::BranchInst>(Terminal)) {
      auto Op = Builder.create<IfOp>(TerminalLoc);

      emitExpressionTree(Op.getCondition(),
                         Branch->getCondition(),
                         TerminalLoc);

      mlir::OpBuilder::InsertionGuard Guard(Builder);

      // Emit true branch:
      revng_assert(Op.getThen().empty());
      Builder.setInsertionPointToEnd(&Op.getThen().emplaceBlock());
      rc_recur emitBasicBlock(Branch->getSuccessor(0), InnerPostDom);

      // Emit false branch:
      revng_assert(Op.getElse().empty());
      Builder.setInsertionPointToEnd(&Op.getElse().emplaceBlock());
      rc_recur emitBasicBlock(Branch->getSuccessor(1), InnerPostDom);
    } else if (auto *Switch = llvm::dyn_cast<llvm::SwitchInst>(Terminal)) {
      llvm::SmallVector<uint64_t> CaseValues;
      CaseValues.reserve(Switch->getNumCases());

      for (auto CH : Switch->cases())
        CaseValues.push_back(CH.getCaseValue()->getZExtValue());

      auto Op = Builder.create<SwitchOp>(TerminalLoc, CaseValues);

      emitExpressionTree(Op.getCondition(),
                         Switch->getCondition(),
                         TerminalLoc);

      mlir::OpBuilder::InsertionGuard Guard(Builder);

      // Emit case blocks:
      for (auto [I, CH] : llvm::enumerate(Switch->cases())) {
        const llvm::BasicBlock *Succ = CH.getCaseSuccessor();

        revng_assert(Op.getCaseRegion(I).empty());
        Builder.setInsertionPointToEnd(&Op.getCaseRegion(I).emplaceBlock());

        if (hasBeenEmitted(Succ))
          emitGoto(TerminalLoc, Succ);
        else
          rc_recur emitBasicBlock(Succ, InnerPostDom);
      }

      // Emit default block:
      if (const llvm::BasicBlock *Succ = Switch->getDefaultDest();
          Succ != nullptr and isScopeGraphEdge(BB, Succ)) {

        revng_assert(Op.getDefaultCaseRegion().empty());
        Builder
          .setInsertionPointToEnd(&Op.getDefaultCaseRegion().emplaceBlock());

        if (hasBeenEmitted(Succ))
          emitGoto(TerminalLoc, Succ);
        else
          rc_recur emitBasicBlock(Succ, InnerPostDom);
      }
    } else {
      revng_abort("Unsupported terminal instruction");
    }
  }

  RecursiveCoroutine<void>
  emitBasicBlock(const llvm::BasicBlock *BB,
                 const llvm::BasicBlock *OuterPostDom) {
    while (BB != OuterPostDom) {
      const auto *InnerPostDom = PostDomTree[BB]->getIDom()->getBlock();
      rc_recur emitBasicBlock(BB, InnerPostDom, OuterPostDom);
      BB = InnerPostDom;
    }
  }

  /* LLVM module import */

  void importFunction(const llvm::Function *F) {
    const model::Function *MF = llvmToModelFunction(Model, *F);

    if (MF == nullptr)
      return;

    auto [Iterator, Inserted] = SymbolMapping.try_emplace(F);
    if (Inserted) {
      auto Type = importModelType(*MF->Prototype());

      Iterator
        ->second = Builder
                     .create<FunctionOp>(getLocation(F->getSubprogram()),
                                         F->getName(),
                                         mlir::cast<clift::FunctionType>(Type));
    }

    auto Op = mlir::cast<FunctionOp>(Iterator->second);
    revng_assert(Op.getArgumentTypes().size() == F->arg_size());

    Op.setHandle(pipeline::locationString(revng::ranks::Function, MF->Entry()));

    revng_assert(Op.getBody().empty());
    mlir::Block &BodyBlock = Op.getBody().emplaceBlock();

    CurrentFunction = Op;
    LabelBlock = &BodyBlock;
    LabelCount = 0;
    LocalCount = 0;

    // Clear the mappings once this function is emitted.
    auto MappingGuard = llvm::make_scope_exit([&]() {
      BlockMapping.clear();
      AllocaMapping.clear();
      ValueMapping.clear();
    });

    for (const auto [A, T] : llvm::zip(F->args(), Op.getArgumentTypes())) {
      mlir::Value Arg = BodyBlock.addArgument(T, Op->getLoc());
      ArgumentMapping.try_emplace(&A, Arg, importLLVMType(A.getType()));
    }

    PostDomTree.recalculate(const_cast<llvm::Function &>(*F));

    mlir::OpBuilder::InsertionGuard BuilderGuard(Builder);
    Builder.setInsertionPointToEnd(&BodyBlock);

    emitBasicBlock(&F->getEntryBlock(), /*OuterPostDom=*/nullptr);
  }

  mlir::OwningOpRef<clift::ModuleOp> importModule(const llvm::Module *Module) {
    auto Op = createOperation<clift::ModuleOp>(Context,
                                               mlir::UnknownLoc::get(Context));

    CurrentModule = Op.get();

    revng_assert(Op->getBody().hasOneBlock());
    Builder.setInsertionPointToEnd(&Op->getBody().front());

    for (const llvm::Function &F : Module->functions())
      importFunction(&F);

    return Op;
  }

  mlir::MLIRContext *const Context;
  const model::Binary &Model;
  mlir::OpBuilder Builder;

  clift::ModuleOp CurrentModule;
  clift::FunctionOp CurrentFunction;

  ScopeGraphPostDomTree PostDomTree;

  uint64_t PointerSize;
  clift::ValueType VoidTypeCache;
  clift::ValueType PointerTypeCache;

  struct BlockMappingInfo {
    // Point where the mapped LLVM IR basic block is emitted. If the block has
    // not yet been visited, this is not set. Note that the LLVM IR basic block
    // is not necessarily emitted at the *start* of any MLIR block.
    mlir::OpBuilder::InsertPoint InsertPoint;

    // The result value of the MakeLabelOp to be used as the target for gotos
    // jumping into the mapped LLVM IR basic block.
    mlir::Value Label;

    // True if the AssignLabelOp has already been created.
    bool HasAssignLabel = false;
  };

  struct ArgumentMappingInfo {
    mlir::Value Argument;
    mlir::Type CastType;
  };

  llvm::DenseMap<const llvm::GlobalObject *, clift::GlobalOpInterface>
    SymbolMapping;
  llvm::DenseMap<llvm::StringRef, clift::FunctionOp> HelperFunctionMapping;
  llvm::DenseMap<const llvm::BasicBlock *, BlockMappingInfo> BlockMapping;
  llvm::DenseMap<const llvm::Argument *, ArgumentMappingInfo> ArgumentMapping;
  llvm::DenseMap<const llvm::AllocaInst *, mlir::Value> AllocaMapping;
  llvm::DenseMap<const llvm::Value *, mlir::Value> ValueMapping;

  mlir::Block *LabelBlock = nullptr;
  unsigned LabelCount = 0;
  unsigned LocalCount = 0;
};

} // namespace

mlir::OwningOpRef<clift::ModuleOp>
clift::importLLVM(mlir::MLIRContext *Context,
                  const model::Binary &Model,
                  const llvm::Module *Module) {
  return LLVMCodeImporter::import(Context, Model, Module);
}
