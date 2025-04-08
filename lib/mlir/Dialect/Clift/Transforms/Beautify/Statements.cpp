//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOpHelpers.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Transforms/Rewrites.h"

namespace clift = mlir::clift;

namespace {

static bool isEmptyRegionOrBlock(mlir::Region &R) {
  return R.empty() or R.front().empty();
}

static bool hasEmptyBlock(mlir::Region &R) {
  return not R.empty() and R.front().empty();
}

// WIP: Merge with the one in CBackend.cpp.
template<typename Operation = mlir::Operation *>
static Operation getOnlyOperation(mlir::Region &R) {
  revng_assert(R.hasOneBlock());
  mlir::Block &B = R.front();
  auto Beg = B.begin();
  auto End = B.end();

  if (Beg == End)
    return {};

  mlir::Operation *Op = &*Beg;

  if (++Beg != End)
    return {};

  if constexpr (std::is_same_v<Operation, mlir::Operation *>) {
    return Op;
  } else {
    return mlir::dyn_cast<Operation>(Op);
  }
}

// WIP: Merge with the one in CliftOps.cpp.
static clift::YieldOp getExpressionYieldOp(mlir::Region &R) {
  if (R.empty())
    return {};

  mlir::Block &B = R.front();

  if (B.empty())
    return {};

  return mlir::dyn_cast<clift::YieldOp>(B.back());
}

static bool isConstantCondition(mlir::Region &R,
                                std::optional<bool> Value = std::nullopt) {
  if (auto Yield = getExpressionYieldOp(R)) {
    if (auto Immediate = Yield.getValue().getDefiningOp<clift::ImmediateOp>())
      return not Value or static_cast<bool>(Immediate.getValue()) == *Value;
  }
  return false;
}

#if 0
class FallthroughRange {
  class sentinel {
  public:
    explicit sentinel() = default;
  };

  class iterator {
  public:
    explicit iterator(mlir::Block::iterator Pos) : Pos(Pos) {}

    mlir::Operation &operator*() const {
      revng_assert(Op != nullptr);
      return *Op;
    }

    iterator &operator++() & {
      revng_assert(Op != nullptr);

      auto Pos = BlockPosition::get(Op);
      auto &[B, I] = Pos;

      while (I == B->end()) {
        mlir::Operation *ParentOp = B->getParentOp();
        if (mlir::isa<clift::FunctionOp, clift::LoopOpInterface>(ParentOp))
          return setEnd();

        
      }

      Op = Pos.getOperation();
      return *this;
    }

    [[nodiscard]] iterator operator++(int) & {
      iterator It = *this;
      ++*this;
      return It;
    }

    friend bool operator==(iterator const& It, sentinel) {
      return Op == nullptr;
    }

  private:
    mlir::Operation *Op;

    iterator &setEnd() {
      Op = nullptr;
      return *this;
    }
  };

public:
  explicit FallthroughRange(mlir::Block *Block, mlir::Block::iterator Pos)
    : Block(Block), Pos(Pos) {}

  iterator begin() const {
    return iterator(Pos != Block->end() ? &*Pos : nullptr);
  }

  sentinel end() const { return sentinel(); }

private:
  mlir::Block *Block;
  mlir::Block::iterator Pos;
};

static FallthroughRange walkOperationsAfter(mlir::Operation *Op) {
  return FallthroughRange(std::next(Op->getIterator()));
}
#endif

[[maybe_unused]] // WIP
static void
removeBlock(mlir::Block *Block) {
  revng_assert(Block->getParent() != nullptr);
  Block->getParent()->getBlocks().remove(Block);
}

template<typename CallableT>
static void replaceExpression(mlir::PatternRewriter &Rewriter,
                              mlir::Region &Region,
                              CallableT &&Callable) {
  auto Yield = getExpressionYieldOp(Region);
  revng_assert(Yield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(Yield.getOperation());

  mlir::Value Value = std::forward<CallableT>(Callable)(Yield.getValue());

  // WIP: Is this assign legal? Should we notify the rewriter somehow?
  Yield->getOpOperand(0).set(Value);
}

template<typename CallableT>
static void mergeExpressionInto(mlir::PatternRewriter &Rewriter,
                                mlir::Region &SourceRegion,
                                mlir::Region &TargetRegion,
                                CallableT &&Callable) {
  auto SourceYield = getExpressionYieldOp(SourceRegion);
  revng_assert(SourceYield);

  auto TargetYield = getExpressionYieldOp(TargetRegion);
  revng_assert(TargetYield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(TargetYield.getOperation());

  mlir::Value Value = std::forward<CallableT>(Callable)(SourceYield.getValue(),
                                                        TargetYield.getValue());

  // WIP: Is this assign legal? Should we notify the rewriter somehow?
  TargetYield->getOpOperand(0).set(Value);
}

static void moveBlocks(mlir::Region &Src, mlir::Region &Dst) {
  Dst.getBlocks().splice(Dst.end(), Src.getBlocks());
}

//===------------------ Future PatternRewriter functions ------------------===//

static void inlineBlockBefore(mlir::Block *Src,
                              mlir::Block *Dst,
                              mlir::Block::iterator Pos) {
  Dst->getOperations().splice(Pos, Src->getOperations());
}

#if 0
static void moveBlockBefore(mlir::PatternRewriter &Rewriter,
                            mlir::Block *Block,
                            mlir::Region *Region,
                            mlir::Region::iterator Pos) {
  revng_assert(Block->getParent() != Region);
  Rewriter.updateRootInPlace(If.getOperation(), [&]() {
    Region->getBlocks().splice(Pos,
                              Block->getParent()->getBlocks(),
                              Block->getIterator());
  });
}
#endif

static mlir::Type getBooleanType(mlir::MLIRContext *Context) {
  return clift::PrimitiveType::get(Context,
                                   clift::PrimitiveKind::SignedKind,
                                   1);
}

[[maybe_unused]] // WIP
static bool
isJumpToStartOf(clift::GoToOp Goto, mlir::Region &Region) {
  if (Region.empty())
    return false;

  for (mlir::Operation &Op : Region.front()) {
    auto AssignLabel = mlir::dyn_cast<clift::AssignLabelOp>(&Op);

    if (not AssignLabel)
      break;

    if (Goto.getLabel() == AssignLabel.getLabel())
      return true;
  }

  return false;
}

struct BlockPosition {
  mlir::Block *Block;
  mlir::Block::iterator Pos;

  static BlockPosition get(mlir::Operation *Op) {
    return BlockPosition{ Op->getBlock(), Op->getIterator() };
  }

  static BlockPosition getNext(mlir::Operation *Op) {
    return BlockPosition{ Op->getBlock(), std::next(Op->getIterator()) };
  }

  static BlockPosition getEnd(mlir::Region &R) {
    return { &R.front(), R.front().end() };
  }

  mlir::Operation *getOperation() const {
    return Block == nullptr or Pos == Block->end() ? nullptr : &*Pos;
  }

  friend bool operator==(BlockPosition const &,
                         BlockPosition const &) = default;
};

static BlockPosition findFallthroughTarget(BlockPosition Position) {
  auto &[B, I] = Position;

  while (I == B->end()) {
    mlir::Operation *ParentOp = B->getParentOp();
    if (mlir::isa<clift::FunctionOp, clift::LoopOpInterface>(ParentOp))
      break;

    Position = BlockPosition::getNext(ParentOp);
  }

  return Position;
}

static BlockPosition getGotoTarget(clift::GoToOp Goto) {
  return BlockPosition::get(Goto.getAssignLabelOp().getOperation());
}

static BlockPosition findJumpTarget(mlir::Operation *Op) {
  if (mlir::isa<clift::ReturnOp>(Op)) {
    auto F = Op->getParentOfType<clift::FunctionOp>();
    return BlockPosition::getEnd(F.getBody());
  }

  if (mlir::isa<clift::LoopContinueOp>(Op)) {
    auto L = Op->getParentOfType<clift::LoopOpInterface>();
    return BlockPosition::getEnd(L.getLoopRegion());
  }

  if (mlir::isa<clift::SwitchBreakOp>(Op)) {
    auto S = Op->getParentOfType<clift::SwitchOp>();
    return findFallthroughTarget(BlockPosition::getNext(S.getOperation()));
  }

  if (auto G = mlir::dyn_cast<clift::GoToOp>(Op))
    return getGotoTarget(G);

  return {};
}

static void invertBooleanExpression(mlir::PatternRewriter &Rewriter,
                                    mlir::Location Loc,
                                    mlir::Region &R) {
  replaceExpression(Rewriter, R, [&](mlir::Value Value) {
    auto BooleanType = getBooleanType(Rewriter.getContext());
    auto Op = Rewriter.create<clift::LogicalNotOp>(Loc,
                                                   BooleanType,
                                                   Value);
    return Op.getResult();
  });
}

static void invertIfStatement(mlir::PatternRewriter &Rewriter, clift::IfOp If) {
  mlir::Region *Then = &If.getThen();
  mlir::Region *Else = &If.getElse();
  revng_assert(not Else->empty());

  invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

  Rewriter.updateRootInPlace(If.getOperation(), [&]() {
    mlir::Block *ThenBlock = Then->empty() ? nullptr : &Then->front();
    mlir::Block *ElseBlock = &Else->front();

    if (ThenBlock != nullptr)
      Then->getBlocks().remove(ThenBlock);

    Else->getBlocks().remove(ElseBlock);
    Then->getBlocks().push_back(ElseBlock);

    if (ThenBlock != nullptr)
      Else->getBlocks().push_back(ThenBlock);
  });
}

//===--------------------- Statement rewrite patterns ---------------------===//

// WIP: Use OpRewritePattern in place of RewritePattern

#if 0
struct NestedIfCombiningPattern : mlir::RewritePattern {
  NestedIfCombiningPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.if", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    auto OuterIf = mlir::cast<clift::IfOp>(Root);
    auto InnerIf = getOnlyOperation<clift::IfOp>(OuterIf.getThen());
    if (not InnerIf) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if does not contain a nested clift.if.";
      });
    }

    mlir::Region &InnerElse = InnerIf.getElse();
    mlir::Region &OuterElse = OuterIf.getElse();

    if (not InnerElse.empty()) {
      if (OuterElse.empty()) {
        return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The inner clift.if has a non-empty else.";
        });
      }

      auto Goto = getOnlyOperation<clift::GoToOp>(InnerElse);
      if (not Goto) {
        return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The inner clift.if does not contain a nested clift.goto.";
        });
      }

      if (not isJumpToStartOf(Goto, OuterElse)) {
        return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The clift.goto does not jump to the outer clift.if else.";
        });
      }
    } else if (not OuterElse.empty()) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The outer clift.if has a non-empty else.";
      });
    }

    mergeExpressionInto(Rewriter,
                        InnerIf.getCondition(),
                        OuterIf.getCondition(),
                        [&](mlir::Value InnerValue, mlir::Value OuterValue) {
      mlir::Location Loc = Rewriter.getFusedLoc(OuterIf.getLoc(),
                                                InnerIf.getLoc());

      auto BooleanType = getBooleanType(Rewriter.getContext());
      auto Op = Rewriter.create<clift::LogicalAndOp>(Loc,
                                                     BooleanType,
                                                     OuterValue,
                                                     InnerValue);

      return Op.getResult();
    });

    mlir::Region &InnerThen = InnerIf.getThen();
    mlir::Region &OuterThen = OuterIf.getThen();
    mlir::Block *InnerThenBlock = &InnerThen.front();

    // WIP: Should we notify the rewriter here?
    removeBlock(InnerThenBlock);
    Rewriter.eraseBlock(&OuterThen.front());
    OuterThen.push_back(InnerThenBlock);

    return mlir::success();
  }
};
#endif

struct LabelCombiningPattern : mlir::RewritePattern {
  LabelCombiningPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.assign_label", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Block::iterator Pos = std::next(Root->getIterator());

    if (Pos == Root->getBlock()->end())
      return mlir::failure();

    auto Label2 = mlir::dyn_cast<clift::AssignLabelOp>(&*Pos);
    if (not Label2)
      return mlir::failure();

    auto Label1 = mlir::cast<clift::AssignLabelOp>(Root);
    Rewriter.replaceAllUsesWith(Label2.getLabel(), Label1.getLabel());
    Rewriter.eraseOp(Label2.getOperation());
    return mlir::success();
  }
};

struct EmptyIfInversionPattern : mlir::RewritePattern {
  EmptyIfInversionPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.if", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    auto If = mlir::cast<clift::IfOp>(Root);

    if (not isEmptyRegionOrBlock(If.getThen())) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has a non-empty then-region or block.";
      });
    }

    if (isEmptyRegionOrBlock(If.getElse())) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has an empty else-region or block.";
      });
    }

    invertIfStatement(Rewriter, If);
    return mlir::success();
  }
};

struct EmptyElseEliminationPattern : mlir::RewritePattern {
  EmptyElseEliminationPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.if", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    auto If = mlir::cast<clift::IfOp>(Root);

    if (not hasEmptyBlock(If.getElse())) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has a non-empty else-region.";
      });
    }

    Rewriter.eraseBlock(&If.getElse().front());
    return mlir::success();
  }
};

struct TerminalIfElseUnwrappingPattern : mlir::RewritePattern {
  TerminalIfElseUnwrappingPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.if", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    auto If = mlir::cast<clift::IfOp>(Root);

    if (isEmptyRegionOrBlock(If.getElse())) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has an empty else-region or block.";
      });
    }

    auto ThenJump = clift::getTrailingJumpOp(If.getThen());
    auto ElseJump = clift::getTrailingJumpOp(If.getElse());

    if (not ThenJump and not ElseJump) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if contains no trailing jumps.";
      });
    }

    if (ThenJump and ElseJump) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if trailing jumps in both branches.";
      });
    }

    if (ElseJump)
      invertIfStatement(Rewriter, If);

    mlir::Block *ElseBlock = &If.getElse().front();
    Rewriter.updateRootInPlace(If.getOperation(), [&]() {
      inlineBlockBefore(ElseBlock,
                        Root->getBlock(),
                        std::next(Root->getIterator()));
    });

    Rewriter.eraseBlock(ElseBlock);
    return mlir::success();
  }
};

struct TrivialJumpEliminationPattern
  : mlir::OpTraitRewritePattern<mlir::OpTrait::clift::NoFallthrough> {
  TrivialJumpEliminationPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    OpTraitRewritePattern(Context, 3) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    if (auto R = mlir::dyn_cast<clift::ReturnOp>(Root)) {
      if (not R.getResult().empty())
        return mlir::failure();
    }

    auto JumpTarget = findJumpTarget(Root);
    auto Fallthrough = findFallthroughTarget(BlockPosition::getNext(Root));

    if (JumpTarget != Fallthrough)
      return mlir::failure();

    Rewriter.eraseOp(Root);
    return mlir::success();
  }
};

struct WhileConditionHoistingPattern : mlir::RewritePattern {
  WhileConditionHoistingPattern(mlir::MLIRContext *Context) :
  // WIP: Think more about the benefit
  RewritePattern("clift.while", 5, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    auto While = mlir::cast<clift::WhileOp>(Root);

    if (not isConstantCondition(While.getCondition(), true))
      return mlir::failure();

    auto If = clift::getLeadingOp<clift::IfOp>(While.getBody());
    if (not If)
      return mlir::failure();

    auto ThenGoto = clift::getTrailingOp<clift::GoToOp>(If.getThen());
    auto ElseGoto = clift::getTrailingOp<clift::GoToOp>(If.getElse());

    if (not ThenGoto or not ElseGoto)
      return mlir::failure();

    auto BreakTarget = BlockPosition::getNext(While);
    bool IsThenBreak = getGotoTarget(ThenGoto) == BreakTarget;
    bool IsElseBreak = getGotoTarget(ElseGoto) == BreakTarget;

    if (IsThenBreak != IsElseBreak)
      return mlir::failure();

    if (IsThenBreak)
      invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

    Rewriter.eraseOp(IsThenBreak ? ThenGoto : ElseGoto);
    mlir::Region &BreakRegion = IsThenBreak ? If.getThen() : If.getElse();
    mlir::Region &OtherRegion = IsThenBreak ? If.getElse() : If.getThen();

    While.getCondition().getBlocks().clear();
    moveBlocks(If.getCondition(), While.getCondition());

    auto &OuterOperations = While->getBlock()->getOperations();
    OuterOperations.splice(std::next(While->getIterator()),
                           BreakRegion.front().getOperations());

    auto &WhileOperations = While.getBody().front().getOperations();
    WhileOperations.splice(std::next(If->getIterator()),
                           OtherRegion.front().getOperations());

    Rewriter.eraseOp(If);

    return mlir::success();
  }
};

struct DoWhileConversionPattern : mlir::RewritePattern {
    DoWhileConversionPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.while", 4, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    auto While = mlir::cast<clift::WhileOp>(Root);

    if (not isConstantCondition(While.getCondition(), true))
      return mlir::failure();

    auto If = clift::getTrailingOp<clift::IfOp>(While.getBody());
    if (not If or not If.getElse().empty())
      return mlir::failure();

    auto Goto = clift::getTrailingOp<clift::GoToOp>(If.getThen());
    if (not Goto or getGotoTarget(Goto) != BlockPosition::getNext(While))
      return mlir::failure();

    // With the break in the true branch, the condition must be inverted.
    invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

    Rewriter.eraseOp(Goto);

    Rewriter.setInsertionPointAfter(While);
    auto DoWhile = Rewriter.create<clift::DoWhileOp>(While.getLoc());

    moveBlocks(If.getCondition(), DoWhile.getCondition());
    moveBlocks(While.getBody(), DoWhile.getBody());

    auto &OuterOperations = While->getBlock()->getOperations();
    OuterOperations.splice(std::next(DoWhile->getIterator()),
                           If.getThen().front().getOperations());

    Rewriter.eraseOp(While);
    Rewriter.eraseOp(If);

    return mlir::success();
  }
};

#if 0
struct LoopDetectionPattern : mlir::RewritePattern {
  LoopDetectionPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.goto", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::RewriteRewriter &Rewriter) const override {
    auto G = mlir::cast<clift::GoToOp>(Root);
    auto L = Goto.getAssignLabelOp();
    mlir::Block *const LB = L->getBlock();

    // Match:
    {
      mlir::Operation *Op = G.getOperation();
      while (Op->getBlock() != LB) {
        mlir::Operation *ParentOp = Op->getParentOp();

        if (not mlir::isa<clift::IfOp, clift::SwitchOp>(ParentOp))
          return mlir::failure();

        Op = ParentOp;
      }

      mlir::Block::iterator Pos = Op->getIterator();
      for (auto I = L->getIterator(), E = LB->end(); I != Pos; ++I) {
        if (I == E)
          return mlir::failure();
      }
    }

    mlir::Operation *Op = G.getOperation();
    while (Op->getBlock() != LB) {
      mlir::Operation *ParentOp = Op->getParentOp();


    }
  }
};
#endif

#if 0
struct OptimizedWhileConversionPattern : mlir::RewritePattern {
  OptimizedWhileConversionPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.do_while", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    
  }
};
#endif

} // namespace

void clift::populateBeautifyStatementRewritePatterns(RewritePatternSet &Set) {
  Set.add(clift::MakeLabelOp::canonicalize);

  Set.add<LabelCombiningPattern>(Set.getContext());
  // Set.add<NestedIfCombiningPattern>(Set.getContext());
  Set.add<EmptyIfInversionPattern>(Set.getContext());
  Set.add<EmptyElseEliminationPattern>(Set.getContext());
  Set.add<TerminalIfElseUnwrappingPattern>(Set.getContext());
  Set.add<TrivialJumpEliminationPattern>(Set.getContext());
  Set.add<DoWhileConversionPattern>(Set.getContext());
}
