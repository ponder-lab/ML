/*
 * Copyright (c) 2018 IBM Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 */
package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.cast.ipa.callgraph.ScopeMappingInstanceKeys.ScopeMappingInstanceKey;
import com.ibm.wala.cast.ir.ssa.AssignInstruction;
import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.Context;
import com.ibm.wala.ipa.callgraph.ContextItem;
import com.ibm.wala.ipa.callgraph.ContextKey;
import com.ibm.wala.ipa.callgraph.ContextSelector;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.ReceiverInstanceContext;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallerSiteContext;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ipa.summaries.SummarizedMethod;
import com.ibm.wala.ipa.summaries.SummarizedMethodWithNames;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSACFG;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.IntSet;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Keeps per-receiver state separate through Python's method-trampoline dispatch (<a
 * href="https://github.com/wala/ML/issues/679">wala/ML#679</a>).
 *
 * <p>All instances of a class share one trampoline method per arity, so a call-string context
 * collapses every receiver into a single trampoline node whose {@code $self} unions the instances —
 * and every method body reached through it (e.g., the Keras lazy {@code build}) unions per-instance
 * state across receivers. Call strings cannot repair this at any depth because they key on the
 * caller's <em>method</em>, not its node.
 *
 * <p>The selector applies four rules, in order, before delegating to the base selector:
 *
 * <ol>
 *   <li>calls made from a synthesized constructor ({@link PythonConstructorFunction}) inherit the
 *       constructor's context, keeping per-construction-site argument values separate (<a
 *       href="https://github.com/wala/ML/issues/671">wala/ML#671</a>);
 *   <li>a trampoline callee ({@link PythonInstanceMethodTrampoline}) is keyed on the dispatched
 *       receiver instance, paired with the calling node and site so distinct call sites of one
 *       instance also stay separate (<a
 *       href="https://github.com/wala/ML/issues/530">wala/ML#530</a>);
 *   <li>the real method body dispatched from a per-receiver trampoline node inherits the
 *       trampoline's context, since a call string would re-collapse it;
 *   <li>any other call made from a receiver-keyed node, or from a node whose own context is a
 *       caller-site pair, is keyed on the calling node and site, so per-caller-node separation
 *       propagates down the helper chains under a receiver context (wala/ML#742) — in particular
 *       the constructors (synthesized or XML-summary) allocating sublayers in the Keras lazy {@code
 *       build}, the summary methods those sublayers dispatch, and the module-level helper functions
 *       sibling layer methods share.
 * </ol>
 *
 * <p>Termination: receiver contexts do not chain unboundedly. Dispatching on a receiver already
 * recorded in the caller's context reuses the caller's context, and {@link #MAX_RECEIVER_DEPTH}
 * caps the caller-pair chain in both rule 2 and rule 4; the cap is load-bearing for rule 4's
 * propagation, which would otherwise grow one pair per call through recursive helpers.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TrampolineReceiverContextSelector implements ContextSelector {

  private static final Logger LOGGER =
      Logger.getLogger(TrampolineReceiverContextSelector.class.getName());

  /**
   * Maximum depth of nested caller-pair contexts before receiver keying degrades to context
   * inheritance. A backstop against unbounded context towers under mutually recursive dispatch
   * across distinct instances; ordinary layer nesting stays far below it.
   */
  private static final int MAX_RECEIVER_DEPTH = 8;

  /** The selector handling every other call. */
  private final ContextSelector base;

  /**
   * Constructs a {@link TrampolineReceiverContextSelector}.
   *
   * @param base The selector handling every other call.
   */
  public TrampolineReceiverContextSelector(ContextSelector base) {
    this.base = base;
  }

  /**
   * The context item naming the attributes the program assigns over the methods a library summary
   * installs on the instance it allocates, before any other use of the instance (wala/ML#1044); a
   * {@link ContextItem.Value} of a set of attribute names. See {@link #shadowedAttributes}.
   */
  public static final ContextKey SHADOWED_ATTRIBUTES =
      new ContextKey() {
        @Override
        public String toString() {
          return "SHADOWED_ATTRIBUTES";
        }
      };

  /**
   * The name under which the summary rewrite also installs a callable's method, for the callable
   * dispatch. It is the analysis's own name, not an attribute a program assigns over.
   */
  private static final String DO_METHOD_NAME = "do";

  /**
   * The method names each summary body installs on what it allocates; see {@link
   * #installedMethods}.
   */
  private final Map<IMethod, Set<String>> installedMethods = HashMapFactory.make();

  /** Whether the program deletes an attribute anywhere; see {@link #programDeletesAttributes}. */
  private Boolean deletesAttributes;

  @Override
  public Context getCalleeTarget(
      CGNode caller, CallSiteReference site, IMethod callee, InstanceKey[] actualParameters) {
    Context context = this.select(caller, site, callee, actualParameters);
    // A library summary whose result the program assigns attributes over before any other use is
    // keyed on those attributes as well, so the instance it allocates there is its own, and the
    // methods it installs under those names are left out of it (wala/ML#1044).
    if (isSummaryBody(callee)) {
      Set<String> shadowed = this.shadowedAttributes(caller, site, callee);
      if (!shadowed.isEmpty()) {
        LOGGER.fine(
            () ->
                "Keying summary "
                    + callee
                    + " at "
                    + site
                    + " in "
                    + caller
                    + " on attributes assigned before any other use: "
                    + shadowed
                    + ".");
        return new ShadowedAttributesContext(context, shadowed);
      }
    }
    return context;
  }

  /**
   * Returns the attributes the program assigns on a call's result before anything else can observe
   * the result (wala/ML#1044). At run time an instance attribute shadows the class's method of that
   * name, and a write that runs before any other use of the instance shadows it for every read, so
   * a method the summary installs under that name is never seen. The writes are the ones that
   * follow the call on its normal path with nothing between them that runs code or uses the result
   * in another way: only other such writes and reads of variables. So the value written is one
   * already held by a variable; a value that must be computed first (a lambda, a call, a read of
   * the result) ends the walk. The method may be seen again only if the attribute is deleted; since
   * the front end does not keep a deleted attribute's name, any deletion in the program disables
   * the rule.
   *
   * <p>Only names the callee installs as methods count: a write of any other name shadows nothing,
   * so the call keeps its context.
   *
   * @param caller The calling node.
   * @param site The call site.
   * @param callee The library summary being called.
   * @return The attribute names, empty if there are none.
   */
  private Set<String> shadowedAttributes(CGNode caller, CallSiteReference site, IMethod callee) {
    IR ir = caller.getIR();
    if (ir == null) return Collections.emptySet();
    SSAAbstractInvokeInstruction[] calls = ir.getCalls(site);
    if (calls.length != 1 || calls[0].getNumberOfReturnValues() == 0) return Collections.emptySet();
    SSAAbstractInvokeInstruction call = calls[0];
    int result = call.getDef();
    SymbolTable symbols = ir.getSymbolTable();
    Set<String> shadowed = new HashSet<>();
    SSACFG cfg = ir.getControlFlowGraph();
    ISSABasicBlock block = ir.getBasicBlockForInstruction(call);
    int index = call.iIndex() + 1;
    SSAInstruction[] instructions = ir.getInstructions();
    while (true) {
      if (index > block.getLastInstructionIndex()) {
        // The call ends its block; continue into its only normal successor, provided the call's
        // block is that successor's only predecessor, so nothing else flows in before the writes.
        Collection<ISSABasicBlock> successors = cfg.getNormalSuccessors(block);
        if (successors.size() != 1) break;
        ISSABasicBlock next = successors.iterator().next();
        if (cfg.getPredNodeCount(next) != 1 || next.isExitBlock()) break;
        block = next;
        index = block.getFirstInstructionIndex();
        continue;
      }
      SSAInstruction instruction = instructions[index++];
      if (instruction == null) continue;
      if (instruction instanceof AstLexicalRead || instruction instanceof AstGlobalRead) continue;
      if (instruction instanceof AstPropertyWrite write
          && write.getObjectRef() == result
          && write.getValue() != result
          && symbols.isStringConstant(write.getMemberRef())) {
        String name = symbols.getStringValue(write.getMemberRef());
        if (!DO_METHOD_NAME.equals(name) && this.installedMethods(callee).contains(name))
          shadowed.add(name);
        continue;
      }
      break;
    }
    if (shadowed.isEmpty() || this.programDeletesAttributes(callee.getClassHierarchy()))
      return Collections.emptySet();
    return shadowed;
  }

  /**
   * The names a library summary installs as methods on what it allocates: the fields it writes with
   * a method trampoline it allocates, which the summary rewrite in {@code PythonAnalysisEngine}
   * adds after the allocation of a class that declares methods. Memoized per summary.
   *
   * @param callee The library summary.
   * @return The installed method names, empty if there are none.
   */
  private Set<String> installedMethods(IMethod callee) {
    return this.installedMethods.computeIfAbsent(
        callee,
        method -> {
          if (!(method instanceof SummarizedMethod summary)) return Collections.emptySet();
          IClassHierarchy cha = method.getClassHierarchy();
          SSAInstruction[] statements = summary.getStatements();
          Set<Integer> trampolines = new HashSet<>();
          for (SSAInstruction statement : statements)
            if (statement instanceof SSANewInstruction allocation
                && cha.lookupClass(allocation.getConcreteType())
                    instanceof PythonInstanceMethodTrampoline) trampolines.add(allocation.getDef());
          if (trampolines.isEmpty()) return Collections.emptySet();
          Set<String> names = new HashSet<>();
          for (SSAInstruction statement : statements)
            if (statement instanceof SSAPutInstruction put
                && !put.isStatic()
                && trampolines.contains(put.getVal()))
              names.add(put.getDeclaredField().getName().toString());
          return names;
        });
  }

  /**
   * Whether any code body of the program calls {@code __delete__}, to which the front end lowers a
   * {@code del} statement, or {@code delattr}. The front end drops the target of a {@code del}, so
   * the deleted attribute's name is unknown and every deletion counts. Computed once, and only when
   * some call's result is assigned attributes before any other use. The scan reads each body's
   * instructions as translated, before SSA conversion, so it neither converts code bodies the
   * analysis never reaches nor depends on their conversion succeeding.
   *
   * @param cha The class hierarchy.
   * @return {@code true} iff the program may delete an attribute.
   */
  private boolean programDeletesAttributes(IClassHierarchy cha) {
    if (this.deletesAttributes == null) {
      boolean deletes = false;
      for (IClass type : cha) {
        if (deletes) break;
        if (!type.getClassLoader().getReference().equals(PythonTypes.pythonLoader)) continue;
        for (IMethod method : type.getDeclaredMethods()) {
          if (method instanceof AstMethod body && callsDeletion(body)) {
            LOGGER.fine("Deletion called in " + body + ".");
            deletes = true;
            break;
          }
        }
      }
      this.deletesAttributes = deletes;
      LOGGER.fine("Program deletes an attribute: " + deletes + ".");
    }
    return this.deletesAttributes;
  }

  /**
   * Whether a code body calls {@code __delete__} or {@code delattr}: an invocation whose function
   * is a value the body defines by allocating that builtin or reading a variable of its name,
   * directly or through assignments. The scan reads the instructions before SSA conversion, where a
   * value may have more than one definition; any definition of a deletion counts, which only
   * disables more. A body with no instructions to read counts as calling one.
   *
   * @param body The code body.
   * @return {@code true} iff the body calls a deletion.
   */
  private static boolean callsDeletion(AstMethod body) {
    // A body without instructions to read cannot be ruled out.
    if (body.cfg() == null) {
      LOGGER.fine(() -> "No instructions to read in " + body + "; counting it as a deletion.");
      return true;
    }
    Object[] instructions = body.cfg().getInstructions();
    Set<Integer> deletions = HashSetFactory.make();
    for (Object instruction : instructions)
      if (instruction instanceof SSANewInstruction allocation) {
        if (isDeletion(allocation.getConcreteType().getName().toString()))
          deletions.add(allocation.getDef());
      } else if (instruction instanceof AstGlobalRead global) {
        if (isDeletion(global.getGlobalName())) deletions.add(global.getDef());
      } else if (instruction instanceof AstLexicalRead lexical)
        for (int i = 0; i < lexical.getAccessCount(); i++)
          if (isDeletion(lexical.getAccess(i).variableName()))
            deletions.add(lexical.getAccess(i).valueNumber());
    if (deletions.isEmpty()) return false;
    // Nothing has propagated copies yet, so follow assignments to a fixpoint, since a call such as
    // `d(o, name)` after `d = delattr` reaches the deletion through the local.
    for (boolean grew = true; grew; ) {
      grew = false;
      for (Object instruction : instructions)
        if (instruction instanceof AssignInstruction assign
            && deletions.contains(assign.getUse(0))
            && deletions.add(assign.getDef())) grew = true;
    }
    for (Object instruction : instructions)
      if (instruction instanceof SSAAbstractInvokeInstruction invoke
          && invoke.getNumberOfUses() > 0
          && deletions.contains(invoke.getUse(0))) return true;
    return false;
  }

  /**
   * Whether a name is that of a deletion builtin. A suffix match also takes names that merely end
   * in these, which only disables more.
   *
   * @param name The name.
   * @return {@code true} iff the name ends in {@code __delete__} or {@code delattr}.
   */
  private static boolean isDeletion(String name) {
    return name != null && (name.endsWith(DELETE_NAME) || name.endsWith(DELATTR_NAME));
  }

  /** The builtin a {@code del} statement is lowered to a call of. */
  private static final String DELETE_NAME = "__delete__";

  /** The builtin that deletes an attribute by name. */
  private static final String DELATTR_NAME = "delattr";

  /**
   * Selects the context before the attribute keying of {@link #getCalleeTarget}.
   *
   * @param caller The calling node.
   * @param site The call site.
   * @param callee The method being called.
   * @param actualParameters The receiver and arguments that are relevant to dispatch.
   * @return The context.
   */
  private Context select(
      CGNode caller, CallSiteReference site, IMethod callee, InstanceKey[] actualParameters) {
    // Calls made from a synthesized constructor inherit its context (wala/ML#671).
    if (caller.getMethod() instanceof PythonConstructorFunction) return caller.getContext();

    // A trampoline callee is keyed on the dispatched receiver instance (wala/ML#679).
    if (callee.getDeclaringClass() instanceof PythonInstanceMethodTrampoline
        && actualParameters != null
        && actualParameters.length > 0
        && actualParameters[0] != null) {
      InstanceKey receiver = actualParameters[0];

      // Recursive dispatch on the caller's own receiver: reuse the caller's context so
      // self-recursive methods do not grow the context.
      if (receiver.equals(caller.getContext().get(ContextKey.RECEIVER))) return caller.getContext();
      // A method object a `super()` object exposes is allocated by the super body once per class
      // and instance, so keying its trampoline on the receiver alone already separates instances;
      // pairing it with the calling node and site as well minted a fresh context per level of a
      // `super().__init__(...)` chain per caller context, and once those bodies ran with a bound
      // `self` everything below the chain multiplied (wala/ML#995). The receiver stays the key,
      // since the trampoline's callee object is filtered to the context's receiver: reusing the
      // caller's context, whose receiver is the instance, left that object empty and the base
      // method never dispatched.
      if (allocatedBySuperBody(receiver)) return new ReceiverInstanceContext(receiver);

      // A Keras layer builds once per instance: `Layer.__call__` runs `build` only while
      // `self.built` is false, and an explicit `layer.build(input_shape)` sets it, so the lazy
      // build a layer-call trampoline injects (wala/ML#595) and the program's explicit call reach
      // ONE build of the instance. Keyed on the receiver alone, a `build` trampoline has one node
      // per instance; keyed on its caller and site as well, it had one per build site per receiver
      // chain, and everything beneath a tower whose layers build their sublayers explicitly
      // doubled with it (wala/ML#1013). The body's `input_shape` then joins the explicit call's
      // shape with the injected call's absence of one, as the one build Keras runs sees whichever
      // site ran it.
      if (isBuildTrampoline(callee)) {
        LOGGER.fine(() -> "Keying build trampoline: " + callee + " on receiver: " + receiver + ".");
        return new ReceiverInstanceContext(receiver);
      }

      // A dispatch to a method already on the caller's chain re-enters it on another receiver: a
      // wrapper layer whose wrapped layer may be a wrapper of its class, as when a field is rebound
      // to a wrapper of its old value (`self.ffn = Wrapper(self.ffn)`) and so holds both. Each
      // re-entry keyed on its caller copied everything beneath it once per level, up to the depth
      // cap, so a stack of such layers multiplied its nodes by the branching at every level. The
      // re-entered method is keyed on the dispatched receiver alone, the per-instance separation
      // the cap below falls back to, as rule 4's recursion guard does for helper chains.
      if (callee.getDeclaringClass() instanceof PythonInstanceMethodTrampoline trampoline
          && dispatchesThroughChain(caller, trampoline.getRealClass())) {
        LOGGER.fine(
            () ->
                "Trampoline re-enters "
                    + trampoline.getRealClass()
                    + " on the chain of caller: "
                    + caller
                    + "; keying on the receiver alone: "
                    + receiver
                    + ".");
        return new ReceiverInstanceContext(receiver);
      }

      if (receiverDepth(caller) >= MAX_RECEIVER_DEPTH) {
        // Past the cap the trampoline is keyed on the dispatched receiver alone. Inheriting the
        // caller's context instead keyed it on the CALLER's receiver, and since a trampoline's
        // callee object is filtered to its context's receiver, the method body was never
        // dispatched from the degraded node (wala/ML#1007).
        LOGGER.fine(
            () ->
                "Receiver-context depth cap reached at caller: "
                    + caller
                    + "; keying on the receiver alone: "
                    + receiver
                    + ".");
        return new ReceiverInstanceContext(receiver);
      }

      LOGGER.fine(() -> "Keying trampoline: " + callee + " on receiver: " + receiver + ".");
      return new AnchoredCallerSiteContext(
          receiverAnchor(caller), caller.getMethod(), site, new ReceiverInstanceContext(receiver));
    }

    // The real method body dispatched from a per-receiver trampoline node stays per-receiver.
    if (caller.getMethod().getDeclaringClass() instanceof PythonInstanceMethodTrampoline)
      return caller.getContext();

    // Calls made from a receiver-keyed method body, or from any node whose own context is a
    // caller-site pair, are keyed on the calling node and site, so per-caller-node separation
    // propagates down the helper chains below a receiver context instead of stopping one hop
    // beneath it (wala/ML#742): a shared module-level helper called by sibling layer methods
    // otherwise collapses onto method-keyed call strings, and every read through it unions the
    // callers' operands (the vendored einsum-via-matmul shim was the witness, minting mixed-rank
    // matmul results from the unioned operands). The keying is the calling node and site alone:
    // pairing in the base selector's call-string context multiplied every keyed node by its
    // base-context tail for ~36% of the layer-heavy whole-project wall clock, with no witness in
    // the suite distinguishing the pair from the bare caller-site key (wala/ML#689). {@link
    // #MAX_RECEIVER_DEPTH} bounds the chain, degrading to the base selector past it; the cap is
    // load-bearing here, since recursive helpers would otherwise grow one pair per recursive
    // call.
    if (caller.getContext().get(ContextKey.RECEIVER) != null
        || caller.getContext() instanceof AnchoredCallerSiteContext) {
      // The `super()` stub is keyed on the super object it is called on, by the selector `super()`
      // installs, so its `$self` read is the one instance that evaluated `super()`. Keyed here on
      // the receiver anchor, the calling method and the site, one stub node served the subclass
      // constructors of every instance built under one anchor, its `$self` read was their union,
      // and the base constructor it reached wrote every instance's arguments to every instance:
      // each wrapper's wrapped-layer field held the other wrappers (wala/ML#1023).
      if (isSuperStub(callee)) return base.getCalleeTarget(caller, site, callee, actualParameters);
      // The guard is recursion, not depth: a deep-but-acyclic layer stack (a BERT encoder tower)
      // legitimately chains many pairs, while a callee already on the caller chain would grow one
      // pair per recursive call. Degrade to the base selector on a method repeat, with an
      // absolute backstop against pathological acyclic towers.
      if (chainsThroughMethod(caller, callee)) {
        LOGGER.fine(
            () ->
                "Caller-pair recursion (or backstop) at caller: "
                    + caller
                    + "; delegating to the base selector for callee: "
                    + callee
                    + ".");
        return base.getCalleeTarget(caller, site, callee, actualParameters);
      }
      // A summary body is keyed on the calling node: keyed on the receiver anchor, the calling
      // method and the site, one summary node served every node of a helper under one anchor, so
      // a `tf.math.log` reached from a `log10` helper's three call sites in one method merged the
      // three arguments, and each site's result carried the others' shapes (wala/ML#716).
      if (isSummaryBody(callee))
        return new AnchoredCallerSiteContext(caller, caller.getMethod(), site, null);
      return new AnchoredCallerSiteContext(receiverAnchor(caller), caller.getMethod(), site, null);
    }
    // A library summary handed a function object is keyed on the calling node and site. Keyed on
    // the site alone, one summary node serves every caller context of that site, and the function
    // values those callers pass merge in it: a decorator built on `functools.wraps` applies the
    // library's pass-through decorator once per decorated function from one site, so the wrappers
    // of every application bound to every decorated name, each wrapper dispatched to every
    // decorated function, and a function received its siblings' positional arguments. The same
    // merge reaches the branches a conditional's summary calls. The function object is recognized
    // at the site, as an argument allocated by a function definition, so no parameter becomes
    // relevant to dispatch. A recursive chain keeps the base key, as the caller-pair rule does.
    if (isSummaryBody(callee)
        && passesFunctionObject(caller, site, callee.getClassHierarchy())
        && !chainsThroughMethod(caller, callee)) {
      LOGGER.fine(
          () -> "Keying summary " + callee + " handed a function object on caller: " + caller);
      return new CallerSiteContext(caller, site);
    }

    return base.getCalleeTarget(caller, site, callee, actualParameters);
  }

  /**
   * Whether a call site passes a function object: an argument, positional or keyword, defined by
   * the allocation of a code body, which is how a function definition or a lambda reaches the site
   * in this front end. A function reaching the site through a variable is not recognized here; it
   * keeps the base key.
   *
   * @param caller The calling node.
   * @param site The call site.
   * @param cha The class hierarchy.
   * @return {@code true} iff some argument at the site is a code-body allocation.
   */
  private static boolean passesFunctionObject(
      CGNode caller, CallSiteReference site, IClassHierarchy cha) {
    IR ir = caller.getIR();
    if (ir == null || caller.getDU() == null) return false;
    IClass codeBody = cha.lookupClass(PythonTypes.CodeBody);
    if (codeBody == null) return false;
    for (SSAAbstractInvokeInstruction call : ir.getCalls(site))
      for (int i = 1; i < call.getNumberOfUses(); i++) {
        SSAInstruction def = caller.getDU().getDef(call.getUse(i));
        if (def instanceof SSANewInstruction allocation) {
          IClass allocated = cha.lookupClass(allocation.getConcreteType());
          if (allocated != null && cha.isSubclassOf(allocated, codeBody)) return true;
        }
      }
    return false;
  }

  /**
   * Whether a callee is a library summary body, a method read from a summary file, as opposed to a
   * program method or one of the Python front end's own synthesized functions (trampolines, super
   * stubs, constructors), which are {@link PythonSummarizedFunction}s.
   *
   * @param callee The method being called.
   * @return {@code true} iff the callee is a summary body.
   */
  private static boolean isSummaryBody(IMethod callee) {
    return callee instanceof SummarizedMethodWithNames
        && !(callee instanceof PythonSummarizedFunction);
  }

  /**
   * Whether a method is the trampoline of a Keras {@code build} method: its declaring class is a
   * trampoline for a method whose class name ends in the build method's name.
   *
   * @param callee The method being called.
   * @return {@code true} iff the method is a {@code build} trampoline.
   */
  private static boolean isBuildTrampoline(IMethod callee) {
    return callee.getDeclaringClass() instanceof PythonInstanceMethodTrampoline trampoline
        && trampoline
            .getRealClass()
            .getReference()
            .getName()
            .toString()
            .endsWith("/" + PythonTypes.KERAS_BUILD_METHOD_NAME);
  }

  /**
   * Returns the nearest node on the given caller's chain whose context is keyed on a receiver: the
   * caller itself when its own context names one, else the first such caller up its caller-site
   * chain, else the caller.
   *
   * <p>Keying a context on the raw calling node makes a node's context its whole call path, since
   * the node's identity carries its own context: the number of contexts of a method is then the
   * number of distinct call paths from a receiver root to it, which multiplies at every fan-in
   * along a layer tower (two fits, a from_config rebuild, a fanout trampoline beside a direct
   * dispatch, the explicit sublayer build beside the lazy-build injection). Anchoring at the
   * receiver-keyed node keeps the separation the rules exist for, per instance and per call site,
   * while a helper chain under one receiver shares that receiver's node.
   *
   * @param caller The calling {@link CGNode}.
   * @return The anchoring node.
   */
  private static CGNode receiverAnchor(CGNode caller) {
    CGNode node = caller;
    while (node.getContext().get(ContextKey.RECEIVER) == null
        && node.getContext() instanceof AnchoredCallerSiteContext chain) node = chain.getAnchor();
    return node.getContext().get(ContextKey.RECEIVER) == null ? caller : node;
  }

  /**
   * Whether a receiver is a method object the {@code super()} body allocates: its allocation site
   * (unwrapped from the scope-mapping key a function object carries) is in a node whose method is
   * the super stub's, declared on the {@code superfun} class.
   *
   * @param receiver A dispatched receiver.
   * @return {@code true} iff the super body allocated it.
   */
  private static boolean allocatedBySuperBody(InstanceKey receiver) {
    InstanceKey key = receiver;
    while (key instanceof ScopeMappingInstanceKey scoped) key = scoped.getBase();
    return key instanceof AllocationSiteInNode allocation
        && allocation
            .getNode()
            .getMethod()
            .getReference()
            .getDeclaringClass()
            .equals(PythonTypes.superfun);
  }

  /**
   * Whether a callee is the {@code super()} stub, the method declared on the {@code superfun}
   * class.
   *
   * @param callee The method being called.
   * @return {@code true} iff the callee is the super stub.
   */
  private static boolean isSuperStub(IMethod callee) {
    return callee.getReference().getDeclaringClass().equals(PythonTypes.superfun);
  }

  @Override
  public IntSet getRelevantParameters(CGNode caller, CallSiteReference site) {
    return base.getRelevantParameters(caller, site);
  }

  /**
   * Returns the depth of the caller-pair context chain rooted at the given node.
   *
   * @param node The node whose context chain to measure.
   * @return The number of {@link CallerSiteContext} links reachable by walking callers from the
   *     given node's context, up to {@link #MAX_RECEIVER_DEPTH}.
   */
  private static int receiverDepth(CGNode node) {
    int depth = 0;
    Context c = node.getContext();
    while (depth < MAX_RECEIVER_DEPTH && c instanceof AnchoredCallerSiteContext) {
      CGNode caller = ((AnchoredCallerSiteContext) c).getAnchor();
      c = caller.getContext();
      depth++;
    }
    return depth;
  }

  /**
   * The absolute bound on rule 4's caller-pair chain length: a backstop against pathological
   * acyclic call towers, far above any real layer stack's nesting.
   */
  private static final int MAX_CALLER_PAIR_DEPTH = 64;

  /**
   * Returns whether keying the given callee on the given caller would recurse: the callee's method
   * already appears on the caller-pair chain (so each recursive call would add a pair), or the
   * chain has reached {@link #MAX_CALLER_PAIR_DEPTH}.
   *
   * @param caller The calling {@link CGNode}.
   * @param callee The dispatched callee.
   * @return {@code true} iff pairing would chain through a repeated method or exceed the backstop.
   */
  private static boolean chainsThroughMethod(CGNode caller, IMethod callee) {
    if (caller.getMethod().equals(callee)) return true; // Direct self-recursion.
    int depth = 0;
    Context c = caller.getContext();
    while (c instanceof AnchoredCallerSiteContext) {
      if (++depth >= MAX_CALLER_PAIR_DEPTH) return true; // Fail closed at the backstop.
      AnchoredCallerSiteContext anchored = (AnchoredCallerSiteContext) c;
      if (anchored.getCallerMethod().equals(callee)) return true; // Already on the chain.
      CGNode up = anchored.getAnchor();
      if (up.getMethod().equals(callee)) return true; // The callee is already on the chain.
      c = up.getContext();
    }
    return allocatesThroughMethod(caller, callee);
  }

  /**
   * Returns whether a trampoline dispatching to the given method class re-enters a method already
   * on the caller's chain: the caller's own method, or a calling method or an anchor's method of
   * the caller's anchored contexts. The class compared is the method's own, the class whose {@code
   * do} body runs it, so distinct layers' {@code call} bodies along a tower never match.
   *
   * @param caller The calling {@link CGNode}.
   * @param methodClass The class of the method the trampoline dispatches to.
   * @return {@code true} iff that method is already on the chain, or the chain reaches {@link
   *     #MAX_CALLER_PAIR_DEPTH}.
   */
  private static boolean dispatchesThroughChain(CGNode caller, IClass methodClass) {
    if (methodClass == null) return false;
    if (caller.getMethod().getDeclaringClass().equals(methodClass)) return true;
    int depth = 0;
    Context c = caller.getContext();
    while (c instanceof AnchoredCallerSiteContext anchored) {
      if (++depth >= MAX_CALLER_PAIR_DEPTH) return true; // Fail closed at the backstop.
      if (anchored.getCallerMethod().getDeclaringClass().equals(methodClass)) return true;
      CGNode up = anchored.getAnchor();
      if (up.getMethod().getDeclaringClass().equals(methodClass)) return true;
      c = up.getContext();
    }
    return false;
  }

  /**
   * The most nodes {@link #allocatesThroughMethod} visits. The contexts share structure, so the
   * walk keeps a visited set and stops here rather than grow with the structure.
   */
  private static final int MAX_CREATOR_VISITS = 64;

  /**
   * Returns whether the given callee already allocated a receiver the caller's context is keyed on,
   * directly or through the contexts of those receivers' own allocating nodes (wala/ML#210).
   *
   * <p>A receiver context nests the context of the node that allocated its receiver, a link the
   * caller chain does not walk. A loop that rebuilds an object under a context keyed on its
   * previous build (a model rebuilt from its own config and stored back where the next round reads
   * it) recurses through that link: each build runs under a context keyed on the last one's
   * receiver, allocates a fresh receiver, and so keys the next round on a context one level deeper,
   * without bound.
   *
   * <p>Two broader bounds break legitimate nesting and should not be re-proposed: a depth cap that
   * counts receiver creators cut encoder towers, whose receivers nest deeply through distinct
   * layers, and a guard on a repeated receiver type dropped nodes from decoder towers, where a
   * shared Keras method object recurs along every nested layer's chain. A repeated allocating
   * method is the existing rule 4 criterion extended by one link, and it leaves those towers
   * unchanged.
   *
   * @param caller The calling {@link CGNode}.
   * @param callee The dispatched callee.
   * @return {@code true} iff the callee appears among the allocating nodes reachable from the
   *     caller's context.
   */
  private static boolean allocatesThroughMethod(CGNode caller, IMethod callee) {
    Set<CGNode> visited = Collections.newSetFromMap(new IdentityHashMap<>());
    Deque<CGNode> pending = new ArrayDeque<>();
    pending.add(caller);
    while (!pending.isEmpty() && visited.size() < MAX_CREATOR_VISITS) {
      CGNode node = pending.poll();
      if (!visited.add(node)) continue;
      Context context = node.getContext();
      if (context.get(ContextKey.RECEIVER) instanceof InstanceKey receiver) {
        CGNode creator = allocator(receiver);
        if (creator != null) {
          if (creator.getMethod().equals(callee)) return true;
          pending.add(creator);
        }
      }
      if (context instanceof AnchoredCallerSiteContext anchored) pending.add(anchored.getAnchor());
    }
    return false;
  }

  /**
   * Returns the node that allocated the given instance, if the instance names one.
   *
   * @param key The instance.
   * @return The allocating node, or {@code null} if the instance names none.
   */
  private static CGNode allocator(InstanceKey key) {
    if (key instanceof ScopeMappingInstanceKey scoped) return scoped.getCreator();
    if (key instanceof AllocationSiteInNode allocation) return allocation.getNode();
    return null;
  }

  /**
   * A context keyed on the nearest receiver-keyed node up the caller chain (the anchor), the method
   * the call site belongs to and the site, with an optional receiver base.
   *
   * <p>The anchor is not the calling node, so this is not a {@link CallerSiteContext}: a reader
   * pairing {@link ContextKey#CALLER} with {@link ContextKey#CALLSITE} would resolve the site's
   * program counter against the wrong method. Several calling nodes with the same method share one
   * instance of this context (the context is interned on its key), so no single calling node
   * exists: {@link ContextKey#CALLER} answers {@code null}, as a context without that information
   * does, rather than an arbitrary representative whose own context would follow solver order;
   * {@link ContextKey#CALLSITE} answers the site, and the calling method and the anchor are
   * available through {@link #getCallerMethod()} and {@link #getAnchor()}. {@link
   * ContextKey#RECEIVER} and the first parameter's filter delegate to the receiver base, so a
   * trampoline's callee object stays filtered to the dispatched instance.
   *
   * <p>The hash code is computed once. WALA recomputes a node's hash from its context's, and a
   * context's from its anchor node's, on every call, so hashing a context walks every anchor
   * reachable through it; caching the hash at every level this selector builds makes each one
   * constant.
   */
  public static final class AnchoredCallerSiteContext implements Context {

    private final CGNode anchor;
    private final IMethod callerMethod;
    private final CallSiteReference site;
    private final Context base;
    private final int hash;

    private AnchoredCallerSiteContext(
        CGNode anchor, IMethod callerMethod, CallSiteReference site, Context base) {
      this.anchor = anchor;
      this.callerMethod = callerMethod;
      this.site = site;
      this.base = base;
      int h = 31 * anchor.hashCode() + callerMethod.hashCode();
      h = 31 * h + site.hashCode();
      this.hash = base == null ? h : 31 * h + base.hashCode();
    }

    /**
     * Returns the nearest receiver-keyed node up the caller chain the context is anchored on.
     *
     * @return The anchor node.
     */
    public CGNode getAnchor() {
      return this.anchor;
    }

    /**
     * Returns the method the context's call site belongs to.
     *
     * @return The calling method.
     */
    public IMethod getCallerMethod() {
      return this.callerMethod;
    }

    /**
     * Returns the context's call site, a program counter within {@link #getCallerMethod()}.
     *
     * @return The call site.
     */
    public CallSiteReference getCallSite() {
      return this.site;
    }

    @Override
    public ContextItem get(ContextKey name) {
      if (name == ContextKey.CALLSITE) return this.site;
      if (name == ContextKey.CALLER) return null; // No single calling node exists; see above.
      return this.base == null ? null : this.base.get(name);
    }

    @Override
    public int hashCode() {
      return this.hash;
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof AnchoredCallerSiteContext other
          && other.hash == this.hash
          && other.anchor.equals(this.anchor)
          && other.callerMethod.equals(this.callerMethod)
          && other.site.equals(this.site)
          && java.util.Objects.equals(other.base, this.base);
    }

    @Override
    public String toString() {
      return "Anchored: "
          + this.anchor
          + " @ "
          + this.callerMethod.getSignature()
          + "@"
          + this.site.getProgramCounter()
          + (this.base == null ? "" : ", Base: " + this.base);
    }
  }

  /**
   * A context that adds the attributes assigned over a summary's installed methods ({@link
   * #SHADOWED_ATTRIBUTES}) to the context the selector would otherwise choose (wala/ML#1044).
   */
  public static final class ShadowedAttributesContext implements Context {

    private final Context base;

    private final Set<String> attributes;

    /**
     * @param base The context the selector would otherwise choose.
     * @param attributes The attribute names.
     */
    private ShadowedAttributesContext(Context base, Set<String> attributes) {
      this.base = base;
      this.attributes = Set.copyOf(attributes);
    }

    @Override
    public ContextItem get(ContextKey name) {
      if (name == SHADOWED_ATTRIBUTES) return new ContextItem.Value<>(this.attributes);
      return this.base.get(name);
    }

    @Override
    public int hashCode() {
      return 31 * this.base.hashCode() + this.attributes.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof ShadowedAttributesContext other
          && this.base.equals(other.base)
          && this.attributes.equals(other.attributes);
    }

    @Override
    public String toString() {
      return this.base + " shadowing " + this.attributes;
    }
  }
}
