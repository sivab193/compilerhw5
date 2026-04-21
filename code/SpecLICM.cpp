/**
 * USC Compiler
 * Jianping Zeng (zeng207@purdue.edu)
 * Speculative Loop Invariant Code Motion (SpecLICM).
*/

#include "Passes.h"
#include "llvm/IR/DataLayout.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/CFG.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Analysis/BranchProbabilityInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/AliasSetTracker.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/PromoteMemToReg.h>
#include <queue>
#include <vector>

using namespace llvm;

bool enableSpecLICM;
namespace {
// Speculative Loop invariant code motion
class SpecLICM : public LoopPass {
public:
  static char ID;
  SpecLICM() : LoopPass(ID) {
    initializeSpecLICMPass(*PassRegistry::getPassRegistry());
  }

  virtual bool runOnLoop(Loop *L, LPPassManager &LPM) override;

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequiredID(LoopSimplifyID);
    AU.addPreservedID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addPreservedID(LCSSAID);

    // Use the built-in Dominator tree and loop info passes
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfo>();
    // Use alias analysis to hoist loads
    AU.addRequired<AliasAnalysis>();
    AU.addPreserved<AliasAnalysis>();
    // Profile-driven frequent-path identification.
    AU.addRequired<BranchProbabilityInfo>();
  }

private:
  // Data regarding the current loop
  Loop *currLoop;
  // The dominator tree for this loop
  DominatorTree *domTree;
  // Loop information for this loop
  LoopInfo *loopInfo;
  // Denotes whether or not loop has been modified
  bool changed;
  AliasSetTracker *aliasSetTracker;
  // Predheader of the current loop
  BasicBlock *preheader;
  BasicBlock *header;
  Function *fn;
  AliasAnalysis *aa;
  const DataLayout *dl;
  DenseMap<Loop *, AliasSetTracker *> loop2AliasSet;
  std::vector<AllocaInst *> aliasAI;
  DenseSet<Instruction*> insertedLds;
  // Profile-driven members. Populated at the start of each
  // runOnLoop invocation from !prof metadata via BranchProbabilityInfo.
  BranchProbabilityInfo *bpi;
  std::set<BasicBlock*> frequentPath;
private:
  void hoistRegion(DomTreeNode *startNode);
  bool isSafeToHoist(Instruction *inst);
  void hoistInst(Instruction *inst);
  bool inCurrentLoop(BasicBlock *bb) {
    return currLoop->contains(bb);
  }
  // Return true if it is safe to hoist this instruction to the preheader.
  bool canHoistInst(Instruction *inst);

  // Speculative hoist loads up to the preheader.
  void specHoistInst(LoadInst *li);

  // Fill up fixup basic block for each speculatively hoisted load.
  void fillFixupBlocks(LoadInst *ld, BasicBlock *fixupBB);

  bool guaranteedToExecute(Instruction *inst);
  void fixPhiNodes(BasicBlock *newHeader);
  void promoteMemToReg();

  // Walk the ≥80% successor chain from the loop header, populating
  // `frequentPath` with the set of blocks on the frequent path. Uses
  // BranchProbabilityInfo::isEdgeHot (threshold built in at >4/5).
  void computeFrequentPath(Loop *L);

  // Return true if any store on the frequent path may-aliases the given
  // load's pointer operand. Syntactic check only (spec assumes no pointer
  // aliasing — we only worry about explicit writes to the same address).
  bool anyConflictOnFrequentPath(LoadInst *ld);

  // Collect the BFS-closure of loop-body users of `li` whose other operands
  // are all loop-invariant (or other chain members). Returns the chain in
  // program (topological) order, ready to be moveBefore'd to the preheader.
  std::vector<Instruction*> collectDependentChain(LoadInst *li);
};
}

char SpecLICM::ID = 0;
INITIALIZE_PASS_BEGIN(SpecLICM, "speclicm", "Speculative Loop Invariant Code Motion", false, false)
  INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
  INITIALIZE_PASS_DEPENDENCY(LCSSA)
  INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(LoopInfo)
  INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfo)
  INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(SpecLICM, "speclicm", "Speculative Loop Invariant Code Motion", false, false)

LoopPass *llvm::createSpecLICMPass() {
  return new SpecLICM();
}

bool SpecLICM::runOnLoop(Loop *L, LPPassManager &LPM) {
  // PA5: Implement if necessary
  changed = false;
  // Save the current loop
  currLoop = L;
  // Grab the loop info
  loopInfo = &getAnalysis<LoopInfo>();
  // Grab the dominator tree
  domTree = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  aa = &getAnalysis<AliasAnalysis>();
  // Grab the branch probability info. If !prof metadata was stamped on the
  // Module by Emitter::doSpecLICM, BPI will reflect it; otherwise it falls
  // back to static heuristics (which won't usually identify an inner-body
  // frequent path, making the profile filter a no-op — exactly the
  // zero-regression fallback we want for plain `-fplicm`).
  bpi = &getAnalysis<BranchProbabilityInfo>();
  computeFrequentPath(L);

  // Get the preheader block to move instructions into...
  preheader = L->getLoopPreheader();
  header = currLoop->getHeader();
  assert(preheader && header && "loop must have preheader and header!");
  fn = header->getParent();

  dl = header->getDataLayout();
  // Collect alias information of subloops.
  aliasSetTracker = new AliasSetTracker(*aa);
  for (Loop::iterator LoopItr = L->begin(), LoopItrE = L->end();
       LoopItr != LoopItrE; ++LoopItr) {
    Loop *subLoop = *LoopItr;
    AliasSetTracker *subLoopAST = loop2AliasSet[subLoop];
    assert(subLoopAST && "must already have alias tracking set for the subloop!");
    aliasSetTracker->add(*subLoopAST);
    delete subLoopAST;
    loop2AliasSet.erase(subLoop);
  }

  // Loop over all basic blocks of the current loop and add them to the alias tracking set.
  // Note that we skip the subloops.
  for (auto I = L->block_begin(), E = L->block_end(); I != E; ++I) {
    if (loopInfo->getLoopFor(*I) == currLoop)
      aliasSetTracker->add(**I);
  }

  // Call hoistRegion function with proper argument
  hoistRegion(domTree->getNode(L->getHeader()));

  // Leverage LLVM's PromoteMemToReg to promote stack-allocated variables to be in SSA form.
  promoteMemToReg();

  // Clear up the information for the next iteration.
  currLoop = nullptr;
  preheader = nullptr;
  aliasAI.clear();
  insertedLds.clear();

  if (L->getParentLoop())
    loop2AliasSet[L] = aliasSetTracker;
  else
    delete aliasSetTracker;
  return changed;
}

// ---------------------------------------------------------------------------
// Helper: syntactic pointer equality (same Value* OR matching GEP structure)
// ---------------------------------------------------------------------------
static bool pointersSyntacticallyEqual(Value *a, Value *b) {
  if (a == b) return true;
  auto *g1 = dyn_cast<GetElementPtrInst>(a);
  auto *g2 = dyn_cast<GetElementPtrInst>(b);
  if (!g1 || !g2) return false;
  if (g1->getPointerOperand() != g2->getPointerOperand()) return false;
  if (g1->getNumIndices() != g2->getNumIndices()) return false;
  for (unsigned i = 0, e = g1->getNumIndices(); i < e; ++i)
    if (g1->getOperand(i + 1) != g2->getOperand(i + 1)) return false;
  return true;
}

// ---------------------------------------------------------------------------
// hoistRegion – pre-order dominator tree walk
// ---------------------------------------------------------------------------
void SpecLICM::hoistRegion(DomTreeNode *startNode) {
  BasicBlock *BB = startNode->getBlock();
  if (inCurrentLoop(BB)) {
    // Cache instructions to avoid invalidation during modification
    std::vector<Instruction*> insts;
    for (BasicBlock::iterator ii = BB->begin(), ie = BB->end(); ii != ie; ++ii)
      insts.push_back(&*ii);
    for (Instruction *I : insts)
      hoistInst(I);
  }
  // Recurse into dominator tree children
  for (unsigned i = 0, e = startNode->getNumChildren(); i < e; ++i)
    hoistRegion(startNode->getChildren()[i]);
}

// ---------------------------------------------------------------------------
// isSafeToHoist – operands loop-invariant, alias set has no Mod for loads
// ---------------------------------------------------------------------------
bool SpecLICM::isSafeToHoist(Instruction *inst) {
  if (!inCurrentLoop(inst->getParent())) return false;
  // All operands must be loop-invariant
  for (unsigned i = 0, e = inst->getNumOperands(); i < e; ++i) {
    if (!currLoop->isLoopInvariant(inst->getOperand(i)))
      return false;
  }
  // For loads: check alias set for modifications
  if (LoadInst *LI = dyn_cast<LoadInst>(inst)) {
    uint64_t sz = dl ? dl->getTypeStoreSize(LI->getType())
                     : AliasAnalysis::UnknownSize;
    AliasSet &AS = aliasSetTracker->getAliasSetForPointer(
        LI->getPointerOperand(), sz,
        LI->getMetadata(LLVMContext::MD_tbaa));
    if (AS.isMod()) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// canHoistInst – safe to hoist AND (guaranteed-to-execute OR safe-to-speculate)
// ---------------------------------------------------------------------------
bool SpecLICM::canHoistInst(Instruction *inst) {
  if (!isSafeToHoist(inst)) return false;
  if (guaranteedToExecute(inst)) return true;
  if (isSafeToSpeculativelyExecute(inst, dl)) return true;
  return false;
}

// ---------------------------------------------------------------------------
// guaranteedToExecute – inst's block dominates all loop exits
// ---------------------------------------------------------------------------
bool SpecLICM::guaranteedToExecute(Instruction *inst) {
  BasicBlock *BB = inst->getParent();
  SmallVector<BasicBlock*, 4> exitBlocks;
  currLoop->getExitBlocks(exitBlocks);
  for (unsigned i = 0, e = exitBlocks.size(); i < e; ++i) {
    if (!domTree->dominates(BB, exitBlocks[i]))
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// hoistInst – decision router: normal hoist or speculative hoist
// ---------------------------------------------------------------------------
void SpecLICM::hoistInst(Instruction *inst) {
  // Try normal hoist first
  if (canHoistInst(inst)) {
    inst->moveBefore(preheader->getTerminator());
    changed = true;
    return;
  }
  // Try speculative hoist for loads
  if (LoadInst *ld = dyn_cast<LoadInst>(inst)) {
    if (enableSpecLICM &&
        currLoop->isLoopInvariant(ld->getPointerOperand()) &&
        !anyConflictOnFrequentPath(ld)) {
      specHoistInst(ld);
    }
  }
}

// ---------------------------------------------------------------------------
// specHoistInst – core speculative LICM logic
// ---------------------------------------------------------------------------
void SpecLICM::specHoistInst(LoadInst *li) {
  assert(enableSpecLICM && "must enable SpecLICM");

  LLVMContext &ctx = fn->getContext();
  Value *loadPtr = li->getPointerOperand();
  Type *loadTy = li->getType();

  // --- A. Collect dependent chain ---
  std::vector<Instruction*> depChain = collectDependentChain(li);

  // --- B. Find all conflicting stores via alias analysis ---
  std::vector<StoreInst*> conflictStores;
  uint64_t loadSz = dl ? dl->getTypeStoreSize(loadTy)
                       : AliasAnalysis::UnknownSize;
  for (Loop::block_iterator bi = currLoop->block_begin(),
       be = currLoop->block_end(); bi != be; ++bi) {
    for (BasicBlock::iterator ii = (*bi)->begin(), ie = (*bi)->end();
         ii != ie; ++ii) {
      StoreInst *si = dyn_cast<StoreInst>(ii);
      if (!si) continue;
      Value *sp = si->getPointerOperand();
      uint64_t storeSz = dl ? dl->getTypeStoreSize(
                                  si->getValueOperand()->getType())
                            : AliasAnalysis::UnknownSize;
      if (aa->alias(AliasAnalysis::Location(loadPtr, loadSz),
                    AliasAnalysis::Location(sp, storeSz))
          != AliasAnalysis::NoAlias) {
        conflictStores.push_back(si);
      }
    }
  }
  if (conflictStores.empty()) return;

  // --- C. Determine pattern ---
  bool allSynMatch = true;
  for (unsigned i = 0; i < conflictStores.size(); ++i) {
    if (!pointersSyntacticallyEqual(loadPtr,
                                    conflictStores[i]->getPointerOperand())) {
      allSynMatch = false;
      break;
    }
  }
  bool useFixup = !depChain.empty() || !allSynMatch;

  // --- D. Move load and chain to preheader ---
  li->moveBefore(preheader->getTerminator());
  for (unsigned i = 0; i < depChain.size(); ++i)
    depChain[i]->moveBefore(preheader->getTerminator());

  Instruction *specVal = depChain.empty() ? static_cast<Instruction*>(li)
                                          : depChain.back();

  // --- E. Create allocas in entry block ---
  Instruction *entryFirst = &*fn->getEntryBlock().getFirstInsertionPt();

  std::string specName = depChain.empty()
      ? std::string(".alloca")
      : (specVal->getName() + ".alloca").str();
  AllocaInst *specAlloca = new AllocaInst(loadTy, specName, entryFirst);
  aliasAI.push_back(specAlloca);

  AllocaInst *aliasAlloca = new AllocaInst(
      Type::getInt1Ty(ctx), "alias", entryFirst);
  aliasAI.push_back(aliasAlloca);

  // --- F. Store initial values in preheader ---
  new StoreInst(specVal, specAlloca, preheader->getTerminator());
  new StoreInst(ConstantInt::getFalse(ctx), aliasAlloca,
                preheader->getTerminator());

  // --- G. Create alias.cmp for each conflicting store ---
  for (unsigned ci = 0; ci < conflictStores.size(); ++ci) {
    StoreInst *si = conflictStores[ci];
    Value *sp = si->getPointerOperand();
    ICmpInst *aliasCmp;

    if (currLoop->isLoopInvariant(sp)) {
      // Move store pointer GEP to preheader if it's inside the loop
      if (Instruction *spI = dyn_cast<Instruction>(sp)) {
        if (currLoop->contains(spI->getParent()))
          spI->moveBefore(preheader->getTerminator());
      }
      aliasCmp = new ICmpInst(preheader->getTerminator(),
                              ICmpInst::ICMP_EQ, loadPtr, sp, "alias.cmp");
    } else {
      // Place alias.cmp before the store instruction
      aliasCmp = new ICmpInst(si, ICmpInst::ICMP_EQ, loadPtr, sp, "alias.cmp");
    }

    // Store alias flag right after the conflicting store instruction
    BasicBlock::iterator afterStore(si);
    ++afterStore;
    new StoreInst(aliasCmp, aliasAlloca, &*afterStore);
  }

  // --- H. Build the header structure ---
  // The "effective header" is what the preheader currently targets
  BasicBlock *effHeader = preheader->getTerminator()->getSuccessor(0);

  if (useFixup) {
    // ========== FIXUP-BRANCH PATTERN ==========

    // H1. Create blocks
    BasicBlock *aliasHeaderBB =
        BasicBlock::Create(ctx, "alias.header", fn, effHeader);
    BasicBlock *aliasFixupBB =
        BasicBlock::Create(ctx, "alias.fixup", fn, effHeader);

    // H2. Redirect preheader → aliasHeaderBB
    preheader->getTerminator()->setSuccessor(0, aliasHeaderBB);

    // H3. Redirect back-edges (latch → aliasHeaderBB instead of effHeader)
    //     Collect predecessors first to avoid iterator invalidation
    SmallVector<BasicBlock*, 4> latches;
    for (pred_iterator pi = pred_begin(effHeader), pe = pred_end(effHeader);
         pi != pe; ++pi) {
      if (currLoop->contains(*pi))
        latches.push_back(*pi);
    }
    for (unsigned i = 0; i < latches.size(); ++i) {
      TerminatorInst *term = latches[i]->getTerminator();
      for (unsigned s = 0; s < term->getNumSuccessors(); ++s) {
        if (term->getSuccessor(s) == effHeader)
          term->setSuccessor(s, aliasHeaderBB);
      }
    }

    // H4. Move existing PHIs from effHeader → aliasHeaderBB
    fixPhiNodes(aliasHeaderBB);

    // H5. Build alias.header body: load alias flag, neg.alias, conditional br
    LoadInst *aliasFlagLD = new LoadInst(aliasAlloca, "", aliasHeaderBB);
    ICmpInst *negAlias = new ICmpInst(*aliasHeaderBB, ICmpInst::ICMP_EQ,
                                      aliasFlagLD, ConstantInt::getFalse(ctx),
                                      "neg.alias");
    BranchInst::Create(effHeader, aliasFixupBB, negAlias, aliasHeaderBB);

    // H6. Fill the fixup block
    BranchInst *fixupBr = BranchInst::Create(effHeader, aliasFixupBB);
    fillFixupBlocks(li, aliasFixupBB);
    // Store fixup results to allocas
    // The last non-terminator in fixupBB is the final recomputed value
    Instruction *fixupVal = &*std::prev(BasicBlock::iterator(fixupBr));
    new StoreInst(fixupVal, specAlloca, fixupBr);
    new StoreInst(ConstantInt::getFalse(ctx), aliasAlloca, fixupBr);

    // H7. In effHeader: load the speculated value and replace uses
    LoadInst *specLD = new LoadInst(specAlloca, "",
                                    &*effHeader->getFirstInsertionPt());
    // Replace uses of specVal within the loop (but not the preheader store)
    SmallVector<Use*, 16> toReplace;
    for (Value::use_iterator ui = specVal->use_begin(),
         ue = specVal->use_end(); ui != ue; ++ui) {
      Instruction *user = dyn_cast<Instruction>(ui->getUser());
      if (user && user != specLD && currLoop->contains(user->getParent()))
        toReplace.push_back(&*ui);
    }
    for (unsigned i = 0; i < toReplace.size(); ++i)
      toReplace[i]->set(specLD);

    // Also store updated alias flag (false) in fixup's stores (already done)
    // Mark the loop as changed
    changed = true;

  } else {
    // ========== SELECT-INLINE PATTERN ==========

    // Rename the effective header to "alias.header"
    effHeader->setName("alias.header");

    // Insert select-based repair at the top of effHeader (after phis)
    Instruction *insertPt = &*effHeader->getFirstInsertionPt();

    // Load alias flag
    LoadInst *aliasFlagLD = new LoadInst(aliasAlloca, "", insertPt);
    // neg.alias = icmp eq alias.0, false
    ICmpInst *negAlias = new ICmpInst(insertPt, ICmpInst::ICMP_EQ,
                                      aliasFlagLD, ConstantInt::getFalse(ctx),
                                      "neg.alias");
    // Fresh reload
    LoadInst *freshLoad = new LoadInst(loadPtr, "", insertPt);
    // Load cached speculated value
    LoadInst *cachedVal = new LoadInst(specAlloca, "", insertPt);

    // select: use cached if no alias, else use fresh
    SelectInst *selVal = SelectInst::Create(negAlias, cachedVal, freshLoad,
                                            "", insertPt);
    // select for alias flag: keep if no alias, else reset to false
    SelectInst *selAlias = SelectInst::Create(
        negAlias, aliasFlagLD, ConstantInt::getFalse(ctx), "", insertPt);

    // Store the selected values back to allocas
    new StoreInst(selVal, specAlloca, insertPt);
    new StoreInst(selAlias, aliasAlloca, insertPt);

    // Replace uses of specVal within the loop
    SmallVector<Use*, 16> toReplace;
    for (Value::use_iterator ui = specVal->use_begin(),
         ue = specVal->use_end(); ui != ue; ++ui) {
      Instruction *user = dyn_cast<Instruction>(ui->getUser());
      if (user && user != selVal && currLoop->contains(user->getParent()))
        toReplace.push_back(&*ui);
    }
    for (unsigned i = 0; i < toReplace.size(); ++i)
      toReplace[i]->set(selVal);

    changed = true;
  }
}

// ---------------------------------------------------------------------------
// fillFixupBlocks – clone load + dependent chain into the fixup block
// ---------------------------------------------------------------------------
void SpecLICM::fillFixupBlocks(LoadInst *ld, BasicBlock *fixupBB) {
  Value *loadPtr = ld->getPointerOperand();
  std::vector<Instruction*> depChain = collectDependentChain(ld);

  // Clone load into fixup block (before terminator)
  LoadInst *reloaded = new LoadInst(loadPtr, "", fixupBB->getTerminator());

  // Clone each chain instruction, remapping operands
  DenseMap<Value*, Value*> remap;
  remap[ld] = reloaded;

  for (unsigned i = 0; i < depChain.size(); ++i) {
    Instruction *clone = depChain[i]->clone();
    clone->setName("");
    for (unsigned o = 0; o < clone->getNumOperands(); ++o) {
      DenseMap<Value*,Value*>::iterator it = remap.find(clone->getOperand(o));
      if (it != remap.end())
        clone->setOperand(o, it->second);
    }
    clone->insertBefore(fixupBB->getTerminator());
    remap[depChain[i]] = clone;
  }
}

// ---------------------------------------------------------------------------
// fixPhiNodes – move PHIs from the old header to the new alias.header
// ---------------------------------------------------------------------------
void SpecLICM::fixPhiNodes(BasicBlock *newHeader) {
  // The "old" header is the block that newHeader will replace as loop entry.
  // We find it as the successor of newHeader's first predecessor's terminator,
  // but actually it's simpler: the old header is the block we are stealing
  // phis FROM. At call-time, the preheader and latches already target
  // newHeader, so the old header is identifiable by the phis whose incoming
  // blocks match newHeader's predecessors.

  // Collect phis from the block that follows newHeader in the function layout
  // (which is the effective header / original header)
  Function::iterator it(newHeader);
  ++it; // skip alias.fixup (inserted between alias.header and effHeader)
  ++it; // this should be effHeader
  BasicBlock *effHeader = &*it;

  // Move all PHI nodes from effHeader to newHeader
  while (PHINode *phi = dyn_cast<PHINode>(&effHeader->front())) {
    phi->moveBefore(newHeader->getTerminator());
  }
}

// ---------------------------------------------------------------------------
// promoteMemToReg – convert alias-tracking allocas to SSA
// ---------------------------------------------------------------------------
void SpecLICM::promoteMemToReg() {
  if (aliasAI.empty()) return;
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  PromoteMemToReg(aliasAI, DT);
}

// ---------------------------------------------------------------------------
// computeFrequentPath – walk hot edges from loop header
// ---------------------------------------------------------------------------
void SpecLICM::computeFrequentPath(Loop *L) {
  frequentPath.clear();
  BasicBlock *BB = L->getHeader();
  while (BB && currLoop->contains(BB)) {
    frequentPath.insert(BB);
    TerminatorInst *term = BB->getTerminator();
    unsigned numSucc = term->getNumSuccessors();
    if (numSucc == 0) break;
    if (numSucc == 1) {
      BB = term->getSuccessor(0);
    } else {
      // Find the hot successor via BranchProbabilityInfo
      BasicBlock *hotSucc = NULL;
      for (unsigned i = 0; i < numSucc; ++i) {
        BasicBlock *succ = term->getSuccessor(i);
        if (bpi->isEdgeHot(BB, succ)) {
          hotSucc = succ;
          break;
        }
      }
      if (!hotSucc) break;  // no hot successor → stop
      BB = hotSucc;
    }
    if (BB == L->getHeader()) break;  // back-edge reached
  }
}

// ---------------------------------------------------------------------------
// anyConflictOnFrequentPath – syntactic check for stores on the hot path
// ---------------------------------------------------------------------------
bool SpecLICM::anyConflictOnFrequentPath(LoadInst *ld) {
  Value *loadPtr = ld->getPointerOperand();
  for (std::set<BasicBlock*>::iterator bi = frequentPath.begin(),
       be = frequentPath.end(); bi != be; ++bi) {
    BasicBlock *BB = *bi;
    for (BasicBlock::iterator ii = BB->begin(), ie = BB->end();
         ii != ie; ++ii) {
      StoreInst *si = dyn_cast<StoreInst>(ii);
      if (si && pointersSyntacticallyEqual(loadPtr, si->getPointerOperand()))
        return true;
    }
  }
  return false;
}

std::vector<Instruction *> SpecLICM::collectDependentChain(LoadInst *li) {
  // BFS over loop-body users of `li`. An instruction joins the chain iff every
  // operand is either (a) the load itself, (b) loop-invariant, or (c) already
  // in the chain. We skip unsupported shapes (loads, stores, phis, terminators)
  // and anything that isn't safe to speculatively execute.
  //
  // Additional safety: if an instruction's value feeds a store whose pointer
  // may-alias the original load's address (a read-modify-write pattern like
  // `X = X + 1`), we must NOT hoist it. Hoisting would cache the add result
  // in the preheader, but the loop body's store would then write that stale
  // cached value instead of `current(X) + 1`. Rejecting these cases keeps the
  // existing SpecLICM read-modify-write semantics intact.
  std::set<Instruction *> inChain;
  std::queue<Instruction *> worklist;

  Value *loadAddr = li->getPointerOperand();
  auto pointerMayAlias = [&](Value *pa, Value *pb) {
    if (pa == pb) return true;
    auto *g1 = dyn_cast<GetElementPtrInst>(pa);
    auto *g2 = dyn_cast<GetElementPtrInst>(pb);
    if (!g1 || !g2) return false;
    if (g1->getPointerOperand() != g2->getPointerOperand()) return false;
    if (g1->getNumIndices() != g2->getNumIndices()) return false;
    for (unsigned i = 0, e = g1->getNumIndices(); i < e; ++i)
      if (g1->getOperand(i + 1) != g2->getOperand(i + 1)) return false;
    return true;
  };
  auto feedsConflictingStore = [&](Instruction *from) {
    for (User *u : from->users()) {
      if (auto *st = dyn_cast<StoreInst>(u)) {
        if (from == st->getValueOperand() &&
            currLoop->contains(st->getParent()) &&
            pointerMayAlias(st->getPointerOperand(), loadAddr)) {
          return true;
        }
      }
    }
    return false;
  };

  auto pushUsersInLoop = [&](Instruction *from) {
    for (User *u : from->users()) {
      if (auto *user = dyn_cast<Instruction>(u)) {
        if (currLoop->contains(user->getParent())) {
          worklist.push(user);
        }
      }
    }
  };
  pushUsersInLoop(li);

  while (!worklist.empty()) {
    Instruction *I = worklist.front();
    worklist.pop();
    if (inChain.count(I)) continue;
    if (isa<LoadInst>(I) || isa<StoreInst>(I) ||
        isa<PHINode>(I) || isa<TerminatorInst>(I)) continue;
    if (!isSafeToSpeculativelyExecute(I, dl)) continue;

    // Reject read-modify-write patterns: if I's value is stored back to an
    // address that may-aliases the original load, hoisting I would cache a
    // stale value in the preheader.
    if (feedsConflictingStore(I)) continue;

    bool ok = true;
    for (Use &u : I->operands()) {
      Value *v = u.get();
      if (v == li) continue;
      if (currLoop->isLoopInvariant(v)) continue;
      if (auto *vi = dyn_cast<Instruction>(v)) {
        if (inChain.count(vi)) continue;
      }
      ok = false;
      break;
    }
    if (!ok) continue;

    inChain.insert(I);
    // Extend the chain transitively: this instruction's users may now also
    // satisfy the "almost invariant" predicate.
    pushUsersInLoop(I);
  }

  // Emit in program (topological) order by walking loop blocks + instructions
  // in their natural sequence, picking up chain members as we go.
  std::vector<Instruction *> ordered;
  for (auto bi = currLoop->block_begin(), be = currLoop->block_end(); bi != be; ++bi) {
    BasicBlock *bb = *bi;
    for (Instruction &I : *bb) {
      if (inChain.count(&I)) ordered.push_back(&I);
    }
  }
  return ordered;
}
