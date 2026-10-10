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

import static com.ibm.wala.cast.python.types.PythonTypes.CALLABLE_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.CALLABLE_METHOD_NAME_FOR_KERAS_MODELS;
import static com.ibm.wala.cast.python.types.PythonTypes.DO_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.INIT_METHOD_NAME;
import static com.ibm.wala.cast.python.util.Util.IMPORT_WILDCARD_CHARACTER;
import static com.ibm.wala.cast.python.util.Util.MODULE_INITIALIZATION_FILENAME;
import static com.ibm.wala.cast.python.util.Util.PYTHON_FILE_EXTENSION;

import com.google.common.collect.Maps;
import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.ipa.callgraph.AstSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.ipa.callgraph.GlobalObjectKey;
import com.ibm.wala.cast.ipa.callgraph.ScopeMappingInstanceKeys.ScopeMappingInstanceKey;
import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.summaries.BuiltinFunctions;
import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.ir.PythonCAstToIRTranslator;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.StarFormalDeclaration;
import com.ibm.wala.cast.python.ssa.ForElementGetInstruction;
import com.ibm.wala.cast.python.ssa.PythonBinaryOpInstruction;
import com.ibm.wala.cast.python.ssa.PythonInstructionVisitor;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.core.util.CancelRuntimeException;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.fixpoint.AbstractOperator;
import com.ibm.wala.fixpoint.UnaryOperator;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.ContextItem;
import com.ibm.wala.ipa.callgraph.ContextKey;
import com.ibm.wala.ipa.callgraph.IAnalysisCacheView;
import com.ibm.wala.ipa.callgraph.propagation.AbstractFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.FilteredPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.FilteredPointerKey.TypeFilter;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKeyFactory;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.StaticFieldKey;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ipa.summaries.SummarizedMethodWithNames;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAArrayLoadInstruction;
import com.ibm.wala.ssa.SSAArrayStoreInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSACFG;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAInvokeInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.Descriptor;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.IntIterator;
import com.ibm.wala.util.intset.IntSetUtil;
import com.ibm.wala.util.intset.MutableIntSet;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

public class PythonSSAPropagationCallGraphBuilder extends AstSSAPropagationCallGraphBuilder {

  /**
   * The synthetic property key under which {@code xs.append(v)} stores {@code v} on {@code xs}; see
   * {@code PythonConstraintVisitor.processListAppend}. Value-iteration surfaces the property's
   * values regardless of its name, so the name only needs to avoid colliding with real program
   * properties.
   */
  public static final String LIST_APPEND_CONTENTS_FIELD = "__list_append_contents__";

  /**
   * The synthetic field holding the elements of a list or tuple produced by repetition ({@code xs *
   * n}) or concatenation ({@code xs + ys}) (wala/ML#960). Like {@value #LIST_APPEND_CONTENTS_FIELD}
   * it is read by every non-constant subscript, iteration and {@code zip}; unlike it, no shape
   * reader interprets it, so a synthesized list, whose length is unknowable, never yields an
   * extent.
   */
  public static final String LIST_OPERATION_CONTENTS_FIELD = "__list_operation_contents__";

  private static final Logger logger =
      Logger.getLogger(PythonSSAPropagationCallGraphBuilder.class.getName());

  public PythonSSAPropagationCallGraphBuilder(
      IClassHierarchy cha,
      AnalysisOptions options,
      IAnalysisCacheView cache,
      PointerKeyFactory pointerKeyFactory) {
    super(PythonLanguage.Python.getFakeRootMethod(cha, cache), options, cache, pointerKeyFactory);
  }

  protected boolean isConstantRef(SymbolTable symbolTable, int valueNumber) {
    return valueNumber != -1 && symbolTable.isConstant(valueNumber);
  }

  @Override
  protected boolean useObjectCatalog() {
    return true;
  }

  @Override
  public GlobalObjectKey getGlobalObject(Atom language) {
    assert language.equals(PythonLanguage.Python.getName());
    return new GlobalObjectKey(cha.lookupClass(PythonTypes.Root));
  }

  @Override
  protected AbstractFieldPointerKey fieldKeyForUnknownWrites(AbstractFieldPointerKey fieldKey) {
    return null;
  }

  @Override
  protected boolean sameMethod(CGNode opNode, String definingMethod) {
    return definingMethod.equals(
        opNode.getMethod().getReference().getDeclaringClass().getName().toString());
  }

  private static final Collection<TypeReference> types =
      Arrays.asList(PythonTypes.string, TypeReference.Int);

  /**
   * A mapping of script names to wildcard imports. We use a {@link Deque} here because we want to
   * always examine the last (front of the queue) encountered wildcard import library for known
   * names assuming that import instructions are traversed from first to last.
   */
  private Map<String, Deque<MethodReference>> scriptToWildcardImports = Maps.newHashMap();

  /**
   * The element types whose slice results get an allocation of their own (wala/ML#916). A subscript
   * with slice syntax lowers to a call of the {@code slice} builtin, whose result used to be its
   * receiver's points-to set, so every reader of the result through the points-to set saw the
   * receiver's pre-slice window. For a receiver key whose concrete type is in this set, the result
   * instead gets a fresh allocation of the type it maps to at the call site; every other key passes
   * through as before. A key maps to a type other than its own where the receiver's class names
   * where it came from rather than what it is: a dataset element's component is a tensor whose
   * class records its index, and its slice is a tensor (wala/ML#1010). Empty by default, so a
   * client that has no element types to name sees no change; the tensor analysis names its tensor
   * and array types. Staged on purpose: only where the element type is known does a slice yield a
   * value of a known kind, so dispatch through the result survives; a general container's subscript
   * yields an element of unknown type and keeps the pass-through.
   */
  private Map<TypeReference, TypeReference> freshSliceResultTypes = Collections.emptyMap();

  /**
   * Names the element types whose slice results get an allocation of their own (wala/ML#916); see
   * {@link #freshSliceResultTypes}.
   *
   * @param types Each concrete receiver type whose slices allocate, mapped to the type of the
   *     slice.
   */
  public void setFreshSliceResultTypes(Map<TypeReference, TypeReference> types) {
    this.freshSliceResultTypes = types == null ? Collections.emptyMap() : Map.copyOf(types);
  }

  /**
   * The receiver instance of the dispatch being resolved, while a target for it is chosen
   * (wala/ML#1012). WALA resolves a dispatching call once per receiver instance but hands the
   * method target selector only the instance's concrete type, and a program instance's concrete
   * type is a plain object: the class it belongs to is recorded only by the constructor that
   * allocated it, which the instance key carries. A selector that dispatches a call on an instance
   * reads the instance here, so the target is decided by that instance alone, never by the callee's
   * points-to set as it stands mid-solve. {@code null} outside a dispatch.
   */
  private InstanceKey dispatchReceiver;

  /**
   * The receiver instance of the dispatch whose target is being chosen; see {@link
   * #dispatchReceiver}.
   *
   * @return The receiver instance, or {@code null} outside a dispatch.
   */
  public InstanceKey getDispatchReceiver() {
    return this.dispatchReceiver;
  }

  @Override
  protected CGNode getTargetForCall(
      CGNode caller, CallSiteReference site, IClass recv, InstanceKey[] iKey) {
    InstanceKey enclosing = this.dispatchReceiver;
    this.dispatchReceiver = site.isDispatch() && iKey != null && iKey.length > 0 ? iKey[0] : null;
    try {
      return super.getTargetForCall(caller, site, recv, iKey);
    } finally {
      this.dispatchReceiver = enclosing;
    }
  }

  /**
   * The array types whose arithmetic yields a fresh array (wala/ML#1009), each mapped to the type
   * of the result: a binary operator with an operand of a key type gets, beside whatever else it
   * produces, a fresh allocation of the mapped type at the operator's instruction index. Without
   * one, {@code x / 255.0} on an array has an empty points-to set, so a tuple, list or field that
   * stores it holds nothing, and every read through that container finds no value although the
   * operator's own result is typed. Empty by default, so a client that names no array types sees no
   * change; the tensor analysis names its tensor and array types.
   */
  private Map<TypeReference, TypeReference> freshBinaryOpResultTypes = Collections.emptyMap();

  /**
   * The length of each {@code *args} pack whose call fixes it: the positional arguments from the
   * {@code *args} formal's index on, when no starred argument can add to them. A pack is allocated
   * at its call rather than by a {@code new} of the caller's IR, so its length is not read off a
   * literal; recorded when the pack is allocated, before its key can reach a reader. A pack a
   * library summary's call allocates is recorded as unknown, {@code -1}, since such a call pads its
   * arguments. Two targets of one call with different {@code *args} indices share one pack, and a
   * length they disagree on is recorded as unknown, {@code -1}; a reader that already read the
   * first target's length keeps it, so the two targets' slices may then miss an element (rare: a
   * polymorphic call whose targets declare {@code *args} at different positions).
   */
  private final Map<InstanceKey, Integer> packLengths = HashMapFactory.make();

  /**
   * The exact leading elements of each tuple concatenation whose left operand is a tuple literal,
   * by the concatenation's node and instruction index: the literal's length. A tuple cannot grow,
   * so in {@code (first,) + rest} element {@code i} below that length is the literal's element
   * {@code i} whatever {@code rest} holds. Recorded when the operator is visited, before its result
   * can be allocated, so a constant subscript never reads a result before its prefix is known.
   */
  private final Map<Pair<CGNode, Integer>, Integer> exactListOperationPrefixes =
      HashMapFactory.make();

  /**
   * Names the array types whose arithmetic allocates a result (wala/ML#1009); see {@link
   * #freshBinaryOpResultTypes}.
   *
   * @param types Each operand type mapped to the type of the result it yields.
   */
  public void setFreshBinaryOpResultTypes(Map<TypeReference, TypeReference> types) {
    this.freshBinaryOpResultTypes = types == null ? Collections.emptyMap() : Map.copyOf(types);
  }

  /**
   * The element types whose element reads get an allocation of their own (wala/ML#1009): an element
   * bound by iterating an array, as the {@code next} a loop is lowered to reads it, or by
   * subscripting an array with an index, is an array of the receiver's kind one rank down. Each
   * concrete receiver type maps to the type of its element. Empty by default; see {@link
   * #setFreshElementTypes}.
   */
  private Map<TypeReference, TypeReference> freshElementTypes = Collections.emptyMap();

  /**
   * Names the element types whose element reads allocate (wala/ML#1009); see {@link
   * #freshElementTypes}.
   *
   * @param types Each concrete receiver type whose element reads allocate, mapped to the type of
   *     the element.
   */
  public void setFreshElementTypes(Map<TypeReference, TypeReference> types) {
    this.freshElementTypes = types == null ? Collections.emptyMap() : Map.copyOf(types);
  }

  /**
   * The attributes the model attaches to each instance of an array type, per attribute name the
   * summary class of the attached method (wala/ML#1009). A fresh array the builder allocates (a
   * slice's or an arithmetic result's) receives them, so it dispatches as an array a summary
   * allocates does. Empty by default.
   */
  private Map<TypeReference, Map<String, TypeReference>> freshArrayAttributes =
      Collections.emptyMap();

  /**
   * Names the per-instance attributes of array types; see {@link #freshArrayAttributes}.
   *
   * @param attributes Each array type mapped to its attributes' summary classes by name.
   */
  public void setFreshArrayAttributes(Map<TypeReference, Map<String, TypeReference>> attributes) {
    this.freshArrayAttributes =
        attributes == null ? Collections.emptyMap() : Map.copyOf(attributes);
  }

  /** The classes already decided by {@link #declaresIterationProtocol}. */
  private final Map<IClass, Boolean> iterationProtocolClasses = HashMapFactory.make();

  /**
   * Whether a class declares Python's iteration protocol, an {@code __iter__} or {@code __next__}
   * method (wala/ML#1010). Iterating an instance of such a class yields what {@code __next__}
   * returns, not the instance's properties.
   *
   * @param type The class.
   * @return {@code true} iff the class or a superclass declares either method.
   */
  public boolean declaresIterationProtocol(IClass type) {
    return iterationProtocolClasses.computeIfAbsent(
        type,
        t -> {
          IClassHierarchy cha = getClassHierarchy();
          for (IClass c = t; c != null; c = c.getSuperclass()) {
            // A summarized or program class renders each method as a function class nested under
            // the class's own name, held by an instance field of the method's name.
            for (String name :
                List.of(BuiltinFunctions.ITER_METHOD_NAME, BuiltinFunctions.NEXT_METHOD_NAME))
              if (cha.lookupClass(
                      TypeReference.findOrCreate(
                          c.getClassLoader().getReference(),
                          TypeName.string2TypeName(c.getName() + "/" + name)))
                  != null) return true;
            java.util.Collection<? extends IMethod> methods = c.getDeclaredMethods();
            if (methods == null) continue;
            for (IMethod m : methods) {
              String name = m.getName().toString();
              if (name.equals(BuiltinFunctions.ITER_METHOD_NAME)
                  || name.equals(BuiltinFunctions.NEXT_METHOD_NAME)) return true;
            }
          }
          return false;
        });
  }

  /** The definer methods and names already scanned for a nested function that writes them. */
  private final Map<Pair<IMethod, String>, Boolean> closureWriters = HashMapFactory.make();

  /**
   * The values of the lexical writes of a variable that reach a closure's allocation in the
   * variable's defining node, when no write can run after the allocation (wala/ML#1026); see {@code
   * PythonConstraintVisitor#visitLexicalReadResolvingCaptures}.
   *
   * <p>The writes are the node's own {@link AstLexicalWrite}s of the name; a write from another
   * nested function (a {@code nonlocal} assignment) is found in that function's own instructions,
   * and its presence declines the whole name. A write is reachable after the allocation when it
   * sits later in the allocation's block or in any block reachable from that block, the
   * allocation's own block included through a back edge, so a rebinding later in a loop body
   * declines. The reaching writes are the standard forward solution over the control-flow graph,
   * with the last write of a block killing the earlier ones; a path from the entry carrying no
   * write at all declines, since the variable may then be unassigned.
   *
   * @param definer The node defining the variable, which created the closure.
   * @param name The variable's name.
   * @param definerName The defining method's name, as the lexical access spells it.
   * @param allocation The instruction index of the closure's allocation in {@code definer}.
   * @return The value numbers written by the reaching writes, or {@code null} to keep the slot.
   */
  Set<Integer> reachingLexicalWrites(
      CGNode definer, String name, String definerName, int allocation) {
    // The solution depends only on the definer's IR, while a closure's every access asks for it
    // once per creation and again on each growth of the closure's function value.
    return reachingLexicalWritesCache
        .computeIfAbsent(
            Pair.make(Pair.make(definer, allocation), Pair.make(name, definerName)),
            key ->
                Optional.ofNullable(
                    computeReachingLexicalWrites(definer, name, definerName, allocation)))
        .orElse(null);
  }

  /** The solutions {@link #reachingLexicalWrites} has computed, a declined one as empty. */
  private final Map<Pair<Pair<CGNode, Integer>, Pair<String, String>>, Optional<Set<Integer>>>
      reachingLexicalWritesCache = HashMapFactory.make();

  private Set<Integer> computeReachingLexicalWrites(
      CGNode definer, String name, String definerName, int allocation) {
    IR ir = definer.getIR();
    if (ir == null) return null;
    if (hasClosureWriter(definer.getMethod(), name, definerName)) return null;
    SSAInstruction[] instructions = ir.getInstructions();
    Map<Integer, Integer> writes = new HashMap<>(); // instruction index -> written value number.
    for (int pc = 0; pc < instructions.length; pc++)
      if (instructions[pc] instanceof AstLexicalWrite write)
        for (Access access : write.getAccesses())
          if (name.equals(access.variableName()) && definerName.equals(access.variableDefiner()))
            writes.put(pc, access.valueNumber());
    if (writes.isEmpty()) return null;
    SSACFG cfg = ir.getControlFlowGraph();
    ISSABasicBlock home = cfg.getBlockForInstruction(allocation);
    if (home == null) return null;
    // No write may run after the allocation: none later in its block, none in a block reachable
    // from it (its own block included, through a back edge).
    for (int pc : writes.keySet())
      if (pc > allocation && cfg.getBlockForInstruction(pc).equals(home)) return null;
    Set<ISSABasicBlock> reachable = new HashSet<>();
    List<ISSABasicBlock> work = new ArrayList<>();
    for (Iterator<ISSABasicBlock> it = cfg.getSuccNodes(home); it.hasNext(); ) work.add(it.next());
    while (!work.isEmpty()) {
      ISSABasicBlock b = work.remove(work.size() - 1);
      if (!reachable.add(b)) continue;
      for (Iterator<ISSABasicBlock> it = cfg.getSuccNodes(b); it.hasNext(); ) work.add(it.next());
    }
    for (int pc : writes.keySet())
      if (reachable.contains(cfg.getBlockForInstruction(pc))) return null;
    // The reaching writes: a block's last write kills the rest; the entry carries the "no write"
    // marker, which declines when it reaches the allocation.
    Map<ISSABasicBlock, Integer> lastWrite = new HashMap<>();
    for (int pc : writes.keySet()) {
      ISSABasicBlock b = cfg.getBlockForInstruction(pc);
      if (b.equals(home) && pc > allocation) continue;
      lastWrite.merge(b, pc, Math::max);
    }
    final int unassigned = -1;
    Map<ISSABasicBlock, Set<Integer>> out = new HashMap<>();
    boolean changed = true;
    while (changed) {
      changed = false;
      for (ISSABasicBlock b : cfg) {
        Set<Integer> in = new HashSet<>();
        if (b.isEntryBlock()) in.add(unassigned);
        for (Iterator<ISSABasicBlock> it = cfg.getPredNodes(b); it.hasNext(); ) {
          Set<Integer> o = out.get(it.next());
          if (o != null) in.addAll(o);
        }
        Set<Integer> o =
            lastWrite.containsKey(b) && !b.equals(home) ? Set.of(lastWrite.get(b)) : in;
        if (!o.equals(out.get(b))) {
          out.put(b, o);
          changed = true;
        }
      }
    }
    Set<Integer> reaching;
    if (lastWrite.containsKey(home)) reaching = Set.of(lastWrite.get(home));
    else {
      reaching = new HashSet<>();
      if (home.isEntryBlock()) reaching.add(unassigned);
      for (Iterator<ISSABasicBlock> it = cfg.getPredNodes(home); it.hasNext(); ) {
        Set<Integer> o = out.get(it.next());
        if (o != null) reaching.addAll(o);
      }
    }
    if (reaching.isEmpty() || reaching.contains(unassigned)) return null;
    Set<Integer> values = new HashSet<>();
    for (int pc : reaching) values.add(writes.get(pc));
    return values;
  }

  /**
   * Whether a function nested in the given method writes the variable: a nested code body with a
   * lexical write of the name under this definer. Decided once per method and name.
   *
   * @param definer The defining method.
   * @param name The variable's name.
   * @param definerName The defining method's name, as a lexical access spells it.
   * @return {@code true} iff a nested function may write the variable.
   */
  private boolean hasClosureWriter(IMethod definer, String name, String definerName) {
    return closureWriters.computeIfAbsent(
        Pair.make(definer, name),
        key -> {
          String prefix = definer.getDeclaringClass().getName().toString() + "/";
          for (IClass c : getClassHierarchy()) {
            if (!c.getName().toString().startsWith(prefix)) continue;
            for (IMethod m : c.getDeclaredMethods()) {
              if (!(m instanceof AstMethod)) continue;
              // The lexical information's read-only set covers only the names an entity itself
              // defines, so a nested function's writes are read off its own instructions.
              IR nested = getAnalysisCache().getIR(m);
              if (nested == null) continue;
              for (SSAInstruction instruction : nested.getInstructions())
                if (instruction instanceof AstLexicalWrite write)
                  for (Access access : write.getAccesses())
                    if (name.equals(access.variableName())
                        && definerName.equals(access.variableDefiner())) return true;
            }
          }
          return false;
        });
  }

  public static class PythonConstraintVisitor extends AstConstraintVisitor
      implements PythonInstructionVisitor {

    /**
     * The None constant has no object catalog for the pointer analysis (wala/ML#964): no field name
     * is recorded on it, so a wildcard read over a set that includes {@code None} enumerates
     * nothing from it. The reflected write null-guards this key; the heap model's catalog key,
     * which consumers read, stays the factory's.
     *
     * @param I The instance key.
     * @return The catalog key, or {@code null} for the None constant.
     */
    @Override
    public PointerKey getPointerKeyForObjectCatalog(InstanceKey I) {
      if (isNoneConstant(I)) return null;
      return super.getPointerKeyForObjectCatalog(I);
    }

    /**
     * A reflected field read on the None constant reads nothing (wala/ML#964).
     *
     * @param I The receiver.
     * @param F The field name key.
     * @return The pointer keys, none for the None constant.
     */
    @Override
    public Iterator<PointerKey> getPointerKeysForReflectedFieldRead(InstanceKey I, InstanceKey F) {
      if (isNoneConstant(I)) return Collections.emptyIterator();
      return super.getPointerKeysForReflectedFieldRead(I, F);
    }

    /**
     * A reflected field write on the None constant writes nothing (wala/ML#964).
     *
     * @param I The receiver.
     * @param F The field name key.
     * @return The pointer keys, none for the None constant.
     */
    @Override
    public Iterator<PointerKey> getPointerKeysForReflectedFieldWrite(InstanceKey I, InstanceKey F) {
      if (isNoneConstant(I)) return Collections.emptyIterator();
      return super.getPointerKeysForReflectedFieldWrite(I, F);
    }

    private static final String GLOBAL_IDENTIFIER = "global";

    private static final Atom IMPORT_FUNCTION_NAME = Atom.findOrCreateAsciiAtom("import");

    @Override
    protected PythonSSAPropagationCallGraphBuilder getBuilder() {
      return (PythonSSAPropagationCallGraphBuilder) builder;
    }

    public PythonConstraintVisitor(AstSSAPropagationCallGraphBuilder builder, CGNode node) {
      super(builder, node);
    }

    /**
     * @param objType the type of the container of which iteration is being done
     * @return whether iteration is over values rather than keys
     *     <p>For some collection types in Python, mainly sets and lists, iteration over a
     *     collection returns the values contained in that collection. For other types, such as
     *     dictionaries, iteration is over the keys that that collection contains.
     *     <p>We also use this mechanism for generators, which put every yielded value in the
     *     synthetic field named {@value PythonTypes#GENERATOR_CONTENT_FIELD_NAME}, which is read by
     *     this mechanism. Generator expressions use the iterator type and generator functions use
     *     CodeBody (wala/ML#696).
     */
    private boolean isValueForKeyType(IClass objType) {
      IClassHierarchy cha = getClassHierarchy();
      return cha.isSubclassOf(objType, cha.lookupClass(PythonTypes.list))
          || cha.isSubclassOf(objType, cha.lookupClass(PythonTypes.set))
          || cha.isSubclassOf(objType, cha.lookupClass(PythonTypes.iterator))
          || cha.isSubclassOf(objType, cha.lookupClass(PythonTypes.CodeBody));
    }

    /**
     * Re-runs the given lexical-access visit whenever a new closure function object reaches this
     * node's function value (value number 1). {@code AstConstraintVisitor.visitLexical} resolves
     * the defining frames of a lexical access from a one-time snapshot of that value's points-to
     * set; when distinct closures of the same function share this node (e.g. under call-string
     * context truncation), a closure arriving after the snapshot never gets its frame wired,
     * silently starving the access and every dispatch downstream of it (<a
     * href="https://github.com/wala/ML/issues/690">wala/ML#690</a>). The side effect re-resolves on
     * every growth of the function value's points-to set; the underlying constraint additions are
     * idempotent, so re-running is safe.
     *
     * @param instruction The lexical access to re-resolve on closure growth.
     * @param revisit Re-invokes the superclass visit for {@code instruction}.
     */
    private void refreshLexicalOnClosureGrowth(SSAInstruction instruction, Runnable revisit) {
      // When the function value's contents are invariant, the visit's snapshot is already
      // complete (the single closure object is known statically) and its points-to set is
      // implicitly represented, so registering a side effect is both unnecessary and disallowed:
      // `newSideEffect` would crash `findOrCreatePointsToSet` (the wala/ML#668 trap; observed as
      // `UnimplementedError` on script-toplevel nodes in WALA's JS test suite for the upstream
      // form of this fix, wala/WALA#1991).
      if (contentsAreInvariant(ir.getSymbolTable(), du, 1)) return;

      PointerKey function = getPointerKeyForLocal(1);
      system.newSideEffect(
          new LexicalRefreshOperator(node, instruction.iIndex(), revisit), function);
    }

    /**
     * The side-effect operator registered by {@link #refreshLexicalOnClosureGrowth}; identity is
     * the (node, instruction index) pair so the fixpoint system de-duplicates re-registrations of
     * the same lexical access.
     */
    private static final class LexicalRefreshOperator extends UnaryOperator<PointsToSetVariable> {
      private final CGNode node;
      private final int instructionIndex;
      private final Runnable revisit;

      private LexicalRefreshOperator(CGNode node, int instructionIndex, Runnable revisit) {
        this.node = node;
        this.instructionIndex = instructionIndex;
        this.revisit = revisit;
      }

      @Override
      public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
        revisit.run();
        return NOT_CHANGED;
      }

      @Override
      public int hashCode() {
        return node.hashCode() * 31 + instructionIndex;
      }

      @Override
      public boolean equals(Object o) {
        return o instanceof LexicalRefreshOperator other
            && instructionIndex == other.instructionIndex
            && node.equals(other.node);
      }

      @Override
      public String toString() {
        return "lexical refresh of " + instructionIndex + " in " + node;
      }
    }

    @Override
    public void visitAstLexicalRead(AstLexicalRead instruction) {
      visitLexicalReadResolvingCaptures(instruction);
      refreshLexicalOnClosureGrowth(
          instruction, () -> visitLexicalReadResolvingCaptures(instruction));
    }

    /**
     * Visits a lexical read, resolving each access that a closure makes of a variable its creator
     * had finished writing before the closure was made, and leaving the rest to the superclass's
     * scope slot (wala/ML#1026). The slot per (name, definer node) receives every write the definer
     * makes, so a closure created after a variable's last assignment read the earlier bindings too:
     * a parameter rebound by a cast before a lambda captured it reached the lambda's callee as both
     * the parameter and the cast. Python binds the variable, not its value, so the closure reads
     * whatever the variable holds when the closure runs; when no write of the name can run after
     * the closure's allocation, that is exactly the set of writes reaching the allocation, and the
     * access is constrained from those writes' values in the creating node.
     *
     * @param instruction The lexical read.
     */
    private void visitLexicalReadResolvingCaptures(AstLexicalRead instruction) {
      List<Access> slot = new ArrayList<>();
      for (Access access : instruction.getAccesses())
        if (!resolveCapturedRead(access)) slot.add(access);
      if (slot.size() == instruction.getAccesses().length) super.visitAstLexicalRead(instruction);
      else if (!slot.isEmpty())
        super.visitAstLexicalRead(
            new AstLexicalRead(instruction.iIndex(), slot.toArray(new Access[0])));
    }

    /**
     * Resolves one lexical access from the writes reaching the closure's allocation in its creating
     * node, when that is sound; see {@link #visitLexicalReadResolvingCaptures}.
     *
     * @param access The access.
     * @return {@code true} iff the access was handled here (constrained, or awaiting the function
     *     value's growth), so the scope slot is not read for it.
     */
    private boolean resolveCapturedRead(Access access) {
      String name = access.variableName();
      String definer = access.variableDefiner();
      if (definer == null || getBuilder().sameMethod(node, definer)) return false;
      List<Pair<CGNode, Integer>> creations = closureCreations(definer);
      if (creations == null) return false; // A function value this rule cannot place: the slot.
      if (creations.isEmpty()) return true; // Nothing has reached the function value yet.
      PointerKey lval = getPointerKeyForLocal(access.valueNumber());
      List<Pair<CGNode, Set<Integer>>> resolved = new ArrayList<>();
      for (Pair<CGNode, Integer> creation : creations) {
        Set<Integer> reaching =
            getBuilder().reachingLexicalWrites(creation.fst, name, definer, creation.snd);
        if (reaching == null) {
          logger.fine(
              () ->
                  "Closure read of "
                      + name
                      + " keeps the slot: a write may follow its creation at "
                      + creation
                      + ".");
          return false;
        }
        resolved.add(Pair.make(creation.fst, reaching));
      }
      for (Pair<CGNode, Set<Integer>> entry : resolved) {
        CGNode creator = entry.fst;
        SymbolTable creatorSymtab =
            getBuilder().getCFAContextInterpreter().getIRView(creator).getSymbolTable();
        DefUse creatorDu = getBuilder().getCFAContextInterpreter().getDU(creator);
        for (int vn : entry.snd) {
          PointerKey rval = getBuilder().getPointerKeyForLocal(creator, vn);
          // An invariant value is represented implicitly, so its contents are added directly, as
          // the superclass's lexical read adds them (the wala/ML#668 trap otherwise).
          if (contentsAreInvariant(creatorSymtab, creatorDu, vn)) {
            system.recordImplicitPointsToSet(rval);
            for (InstanceKey ik : getInvariantContents(creatorSymtab, creatorDu, creator, vn)) {
              system.findOrCreateIndexForInstanceKey(ik);
              system.newConstraint(lval, ik);
            }
          } else system.newConstraint(lval, assignOperator, rval);
        }
      }
      return true;
    }

    /**
     * Whether a method is a module's body: a script's code body, whose class is the script itself
     * rather than a function nested under it. The script's class is named after the script's path,
     * so a module under a directory carries a slash in its name as a nested function does; the path
     * alone ends in the script's {@code .py} suffix, since a function, class or method body nested
     * in the script appends {@code /name} to it.
     *
     * @param method The method.
     * @return {@code true} iff the method is a script body.
     */
    private static boolean isModuleBody(IMethod method) {
      String name = method.getDeclaringClass().getName().toString();
      return name.startsWith("Lscript ") && name.endsWith(".py");
    }

    /**
     * The closures this node runs as, each as its creating node and the allocation's instruction
     * index, when every function value is a closure the given definer created directly.
     *
     * @param definer The name of the method defining the variable read.
     * @return The creations; empty when the function value holds nothing yet; {@code null} when a
     *     function value is not a closure the definer itself created.
     */
    private List<Pair<CGNode, Integer>> closureCreations(String definer) {
      List<Pair<CGNode, Integer>> ret = new ArrayList<>();
      SymbolTable symtab = ir.getSymbolTable();
      List<InstanceKey> functions = new ArrayList<>();
      if (contentsAreInvariant(symtab, du, 1)) {
        for (InstanceKey ik : getInvariantContents(symtab, du, node, 1)) functions.add(ik);
      } else {
        PointsToSetVariable value = system.findOrCreatePointsToSet(getPointerKeyForLocal(1));
        if (value.getValue() != null)
          value.getValue().foreach(i -> functions.add(system.getInstanceKey(i)));
      }
      for (InstanceKey function : functions) {
        if (!(function instanceof ScopeMappingInstanceKey closure)) return null;
        AllocationSiteInNode site;
        try {
          site = com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode(closure.getBase());
        } catch (IllegalArgumentException e) {
          return null;
        }
        if (site == null || !getBuilder().sameMethod(site.getNode(), definer)) return null;
        // A module's variables have feeders beyond the module's own writes: the builtins and the
        // imports bound into its scope, and `global` statements elsewhere. The module keeps the
        // slot; the rule serves a function's locals.
        if (isModuleBody(site.getNode().getMethod())) return null;
        ret.add(Pair.make(site.getNode(), site.getSite().getProgramCounter()));
      }
      return ret;
    }

    @Override
    public void visitAstLexicalWrite(AstLexicalWrite instruction) {
      super.visitAstLexicalWrite(instruction);
      refreshLexicalOnClosureGrowth(instruction, () -> super.visitAstLexicalWrite(instruction));
    }

    @Override
    public void visitForElementGet(ForElementGetInstruction forElementGet) {
      SymbolTable symtab = ir.getSymbolTable();
      int objVn = forElementGet.getUse(0);
      final PointerKey objKey = getPointerKeyForLocal(objVn);
      int eltVn = forElementGet.getUse(1);
      final PointerKey eltKey = getPointerKeyForLocal(eltVn);
      int resultVn = forElementGet.getDef();
      final PointerKey resultKey = getPointerKeyForLocal(resultVn);

      if (contentsAreInvariant(symtab, du, objVn)) {
        for (InstanceKey ik : getInvariantContents(objVn)) {
          if (!isValueForKeyType(ik.concreteType())) {
            system.newConstraint(resultKey, assignOperator, eltKey);
          } else {
            newFieldRead(node, objVn, eltVn, resultVn);
          }
        }
      } else {
        system.newConstraint(
            resultKey,
            new AbstractOperator<PointsToSetVariable>() {
              @Override
              public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable[] rhs) {
                boolean changed = false;
                for (PointsToSetVariable rv : rhs) {
                  if (rv.getValue() != null) {
                    IntIterator is = rv.getValue().intIterator();
                    while (is.hasNext()) {
                      InstanceKey ik = system.getInstanceKey(is.next());
                      if (!isValueForKeyType(ik.concreteType())) {
                        changed |= system.newConstraint(resultKey, assignOperator, eltKey);
                      } else {
                        newFieldRead(node, objVn, eltVn, resultVn);
                      }
                    }
                  }
                }
                if (changed) {
                  return CHANGED;
                } else {
                  return NOT_CHANGED;
                }
              }

              @Override
              public int hashCode() {
                return objKey.hashCode() * eltKey.hashCode();
              }

              @Override
              public boolean equals(Object o) {
                return this == o;
              }

              @Override
              public String toString() {
                return "next element of " + objKey;
              }
            },
            objKey,
            eltKey);
      }
    }

    @Override
    public void visitGet(SSAGetInstruction instruction) {
      SymbolTable symtab = ir.getSymbolTable();
      String name = instruction.getDeclaredField().getName().toString();

      int objVn = instruction.getRef();
      final PointerKey objKey = getPointerKeyForLocal(objVn);

      int lvalVn = instruction.getDef();
      final PointerKey lvalKey = getPointerKeyForLocal(lvalVn);

      if (contentsAreInvariant(symtab, du, objVn)) {
        system.recordImplicitPointsToSet(objKey);
        for (InstanceKey ik : getInvariantContents(objVn)) {
          if (types.contains(ik.concreteType().getReference())) {
            @SuppressWarnings("unused")
            Pair<String, TypeReference> key = Pair.make(name, ik.concreteType().getReference());
            // system.newConstraint(lvalKey, new ConcreteTypeKey(getBuilder().ensure(key)));
          }
        }
      } else {
        system.newSideEffect(
            new AbstractOperator<PointsToSetVariable>() {
              @Override
              public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable[] rhs) {
                if (rhs[0].getValue() != null)
                  rhs[0]
                      .getValue()
                      .foreach(
                          (i) -> {
                            InstanceKey ik = system.getInstanceKey(i);
                            if (types.contains(ik.concreteType().getReference())) {
                              @SuppressWarnings("unused")
                              Pair<String, TypeReference> key =
                                  Pair.make(name, ik.concreteType().getReference());
                              // system.newConstraint(lvalKey, new
                              // ConcreteTypeKey(getBuilder().ensure(key)));
                            }
                          });
                return NOT_CHANGED;
              }

              @Override
              public int hashCode() {
                return node.hashCode() * instruction.hashCode();
              }

              @Override
              public boolean equals(Object o) {
                return getClass().equals(o.getClass()) && hashCode() == o.hashCode();
              }

              @Override
              public String toString() {
                return "get function " + name + " at " + instruction;
              }
            },
            new PointerKey[] {lvalKey});
      }

      // TODO Auto-generated method stub
      super.visitGet(instruction);
    }

    @Override
    public void visitPythonInvoke(PythonInvokeInstruction inst) {
      visitInvokeInternal(inst, new DefaultInvariantComputer());
      processListAppend(inst);
      processTextRead(inst);
      processDictMethod(inst);
      processNamedTupleConstruction(inst);
    }

    /**
     * Constructs a {@code collections.namedtuple} instance where a call's callee may be a
     * namedtuple type, by a side effect on the callee's points-to set; see {@link
     * NamedTupleOperator}. A callee whose keys are invariant or implicitly represented is read
     * directly, since a side effect must not materialize its key (the wala/ML#668 trap).
     *
     * @param inst The call.
     */
    private void processNamedTupleConstruction(PythonInvokeInstruction inst) {
      if (!inst.hasDef()) return;
      SymbolTable symtab = ir.getSymbolTable();
      // A starred argument spreads elements whose positions this binding does not read, so the
      // positions from it on are left unbound rather than bound to the iterable itself.
      int starred = inst.firstStarredPosition();
      int positional =
          starred >= 1
              ? Math.min(starred, inst.getNumberOfPositionalParameters())
              : inst.getNumberOfPositionalParameters();
      Object[] positionalArguments = new Object[Math.max(0, positional - 1)];
      for (int i = 1; i < positional; i++)
        positionalArguments[i - 1] = argumentValue(symtab, inst.getUse(i));
      Map<String, Object> keywordArguments = new LinkedHashMap<>();
      if (inst.getKeywords() != null)
        for (String keyword : inst.getKeywords())
          keywordArguments.put(keyword, argumentValue(symtab, inst.getUse(keyword)));
      NamedTupleOperator operator =
          getBuilder()
          .new NamedTupleOperator(
              node,
              inst.iIndex(),
              getPointerKeyForLocal(inst.getDef()),
              positionalArguments,
              keywordArguments);
      int calleeVn = inst.getUse(0);
      PointerKey calleeKey = getPointerKeyForLocal(calleeVn);
      if (contentsAreInvariant(symtab, du, calleeVn) || system.isImplicit(calleeKey)) {
        for (InstanceKey key : getInvariantContents(symtab, du, node, calleeVn))
          operator.construct(key);
        return;
      }
      system.newSideEffect(operator, calleeKey);
    }

    /**
     * An argument's value as a namedtuple construction binds it: its keys when they are invariant,
     * its pointer key otherwise.
     *
     * @param symtab The symbol table.
     * @param vn The argument's value number.
     * @return An {@code InstanceKey[]} or a {@link PointerKey}.
     */
    private Object argumentValue(SymbolTable symtab, int vn) {
      PointerKey key = getPointerKeyForLocal(vn);
      return contentsAreInvariant(symtab, du, vn) || system.isImplicit(key)
          ? getInvariantContents(symtab, du, node, vn)
          : key;
    }

    /**
     * Reads a negative constant subscript of a tuple as the element it denotes, by a side effect on
     * the subscripted object's points-to set (wala/ML#988). A literal or otherwise implicitly
     * represented object is read directly, since a side effect must not materialize its key (the
     * wala/ML#668 trap).
     *
     * @param instruction The property read.
     */
    private void processNegativeSubscript(AstPropertyRead instruction) {
      SymbolTable symtab = ir.getSymbolTable();
      int memberRef = instruction.getMemberRef();
      // A Python integer literal is a `Long` constant; a string key is an attribute name.
      if (!symtab.isConstant(memberRef)
          || !(symtab.getConstantValue(memberRef) instanceof Long member)
          || member >= 0
          || member < Integer.MIN_VALUE) return;
      int index = member.intValue();
      NegativeSubscriptOperator operator =
          getBuilder()
          .new NegativeSubscriptOperator(getPointerKeyForLocal(instruction.getDef()), index);
      int objectVn = instruction.getObjectRef();
      PointerKey objectKey = getPointerKeyForLocal(objectVn);
      if (contentsAreInvariant(symtab, du, objectVn) || system.isImplicit(objectKey)) {
        for (InstanceKey key : getInvariantContents(symtab, du, node, objectVn)) operator.read(key);
        return;
      }
      system.newSideEffect(operator, objectKey);
    }

    /**
     * Reads a subscript of a list or tuple by a loop variable, {@code xs[i]} inside {@code for i in
     * ...}, as any of the collection's elements (wala/ML#993). The ordinary read names the field by
     * the index's points-to set, and the loop variable of {@code for i in range(n)} has none (the
     * elements of {@code range} are not modeled as integers), so the read was empty, and a call
     * through it, such as {@code self.subnets[i](x)}, reached nothing. Iteration already reads a
     * collection this way, through its catalog of element keys; this does the same for an indexed
     * read. Only a loop variable triggers it: an index bound to a constant some other way (a local
     * or a parameter fed a literal) keeps the exact ordinary read, and a receiver that is neither a
     * list nor a tuple contributes nothing.
     *
     * @param instruction The property read.
     */
    private void processUnknownIndexRead(AstPropertyRead instruction) {
      SymbolTable symtab = ir.getSymbolTable();
      if (symtab.isConstant(instruction.getMemberRef())
          || !isLoopVariable(instruction.getMemberRef())) return;
      UnknownIndexReadOperator operator =
          getBuilder()
          .new UnknownIndexReadOperator(
              getPointerKeyForLocal(instruction.getDef()), instruction.iIndex());
      int objectVn = instruction.getObjectRef();
      PointerKey objectKey = getPointerKeyForLocal(objectVn);
      if (contentsAreInvariant(symtab, du, objectVn) || system.isImplicit(objectKey)) {
        for (InstanceKey key : getInvariantContents(symtab, du, node, objectVn)) operator.read(key);
        return;
      }
      system.newSideEffect(operator, objectKey);
    }

    /**
     * Whether the property read being visited is an element read; see {@link #visitPropertyRead}.
     */
    private boolean elementRead = false;

    @Override
    protected ReflectedFieldAction fieldReadAction(PointerKey lhs) {
      ReflectedFieldAction read = super.fieldReadAction(lhs);
      if (!elementRead) return read;
      return new ReflectedFieldAction() {
        @Override
        public void dump(AbstractFieldPointerKey fieldKey, boolean constObj, boolean constProp) {
          read.dump(fieldKey, constObj, constProp);
        }

        @Override
        public void action(AbstractFieldPointerKey fieldKey) {
          // An array's properties are the methods its model attaches per instance, not its
          // elements: its element is the allocation the read makes (wala/ML#1009), and reading the
          // methods as elements made the loop variable the array's methods.
          IClass type = fieldKey.getInstanceKey().concreteType();
          if (!getBuilder().declaresIterationProtocol(type)
              && !getBuilder().freshElementTypes.containsKey(type.getReference()))
            read.action(fieldKey);
        }
      };
    }

    /**
     * Whether a value is an element read: a read of a collection keyed by one of its property
     * names, {@code e = coll[name]} with {@code name} drawn by an {@link
     * EachElementGetInstruction}, as {@code next} reads a sequence's elements and a comprehension's
     * machinery reads its iterables (wala/ML#1010).
     *
     * @param vn The value number.
     * @return {@code true} iff the value is defined by such a read.
     */
    private boolean isLoopVariable(int vn) {
      SSAInstruction def = du.getDef(vn);
      if (def instanceof AstPropertyRead read
          && du.getDef(read.getMemberRef()) instanceof EachElementGetInstruction) return true;
      // A `for` loop binds its variable to `next` of the iterator `iter` made (wala/ML#1010).
      if (!(def instanceof PythonInvokeInstruction call) || call.getNumberOfUses() < 2)
        return false;
      SSAInstruction callee = du.getDef(call.getUse(0));
      if (callee instanceof AstLexicalRead lexical)
        for (AstLexicalAccess.Access access : lexical.getAccesses())
          if (NEXT_BUILTIN_NAME.equals(access.variableName())) return true;
      return callee instanceof AstGlobalRead global
          && global.getGlobalName().equals("global " + NEXT_BUILTIN_NAME);
    }

    /** The name a program calls the {@code next} builtin by. */
    private static final String NEXT_BUILTIN_NAME = "next";

    /**
     * Surfaces append-accumulated list contents at subscript reads (<a
     * href="https://github.com/wala/ML/issues/661">wala/ML#661</a>): a property read whose member
     * is not a constant string also reads the synthetic {@value #LIST_APPEND_CONTENTS_FIELD}
     * property, the read-side dual of {@link #processListAppend}'s write. Without this, values
     * accumulated through {@code append} surface only under value iteration, and an indexed
     * dispatch such as {@code self.sub_layers[i](x)} over an append-built list has an empty callee
     * set. Named-attribute reads (constant string members) are excluded so method and attribute
     * lookups do not observe element values; the synthetic property is only ever written on append
     * receivers, so on all other objects the extra read contributes nothing.
     *
     * @param instruction The property read to examine.
     */
    private void processListContentsRead(AstPropertyRead instruction) {
      SymbolTable symtab = ir.getSymbolTable();
      int memberRef = instruction.getMemberRef();
      if (symtab.isConstant(memberRef) && symtab.getConstantValue(memberRef) instanceof String)
        return;

      // A constant index below a concatenation's exact prefix names the prefix element alone,
      // which the ordinary read takes from its numbered field; the order-free contents are read
      // only by an index the prefix does not cover.
      if (symtab.isConstant(memberRef)
          && (symtab.getConstantValue(memberRef) instanceof Long
              || symtab.getConstantValue(memberRef) instanceof Integer)) {
        long index = ((Number) symtab.getConstantValue(memberRef)).longValue();
        if (index >= 0 && index <= Integer.MAX_VALUE) {
          ConstantIndexContentsReadOperator operator =
              getBuilder()
              .new ConstantIndexContentsReadOperator(
                  getPointerKeyForLocal(instruction.getDef()), (int) index);
          int objectVn = instruction.getObjectRef();
          PointerKey objectKey = getPointerKeyForLocal(objectVn);
          if (contentsAreInvariant(symtab, du, objectVn) || system.isImplicit(objectKey)) {
            for (InstanceKey key : getInvariantContents(symtab, du, node, objectVn))
              operator.read(key);
            return;
          }
          system.newSideEffect(operator, objectKey);
          return;
        }
      }

      InstanceKey contentsKey =
          getBuilder().getInstanceKeyForConstant(PythonTypes.string, LIST_APPEND_CONTENTS_FIELD);
      InstanceKey operationKey =
          getBuilder().getInstanceKeyForConstant(PythonTypes.string, LIST_OPERATION_CONTENTS_FIELD);

      newFieldOperationFieldConstant(
          node,
          true,
          fieldReadAction(getPointerKeyForLocal(instruction.getDef())),
          instruction.getObjectRef(),
          new InstanceKey[] {contentsKey, operationKey});
    }

    /**
     * Models {@code xs.append(v)} as a property write of {@code v} onto {@code xs} under the
     * synthetic {@value #LIST_APPEND_CONTENTS_FIELD} key, so values accumulated through {@code
     * append} surface when the collection is iterated ({@code visitForElementGet} reads the
     * cataloged properties of value-iterated collections). Nothing else models {@code list.append},
     * so without this the appended values are unreachable from the collection and every value
     * flowing through an append-accumulate-iterate chain unravels (wala/ML#570, wala/ML#618).
     *
     * <p>Detection is syntactic on the du-chain: an invoke whose callee value is a property read of
     * the constant name {@code append}. The property write coexists with the invoke's normal
     * dispatch, so a user-defined {@code append} method still dispatches; such a receiver merely
     * gains a stray property under the synthetic key, which is only observable if that same object
     * is also value-iterated.
     *
     * @param inst the invoke instruction to examine
     */
    private void processListAppend(PythonInvokeInstruction inst) {
      if (inst.getNumberOfPositionalParameters() != 2) return;

      SSAInstruction calleeDef = du.getDef(inst.getUse(0));
      if (!(calleeDef instanceof AstPropertyRead)) return;

      AstPropertyRead read = (AstPropertyRead) calleeDef;
      SymbolTable symtab = ir.getSymbolTable();
      if (!symtab.isConstant(read.getMemberRef())
          || !"append".equals(symtab.getConstantValue(read.getMemberRef()))) return;

      InstanceKey contentsKey =
          getBuilder().getInstanceKeyForConstant(PythonTypes.string, LIST_APPEND_CONTENTS_FIELD);
      int valueVn = inst.getUse(1);

      if (contentsAreInvariant(symtab, du, valueVn)) {
        // The invoke's own argument processing records an invariant value's pointer key as
        // implicitly represented, so a raw constraint on that key crashes
        // `findOrCreatePointsToSet` (wala/ML#668). Write the invariant instance keys directly.
        system.recordImplicitPointsToSet(getPointerKeyForLocal(valueVn));
        newFieldWrite(
            node,
            read.getObjectRef(),
            new InstanceKey[] {contentsKey},
            getInvariantContents(symtab, du, node, valueVn));
      } else
        newFieldWrite(
            node,
            read.getObjectRef(),
            new InstanceKey[] {contentsKey},
            getPointerKeyForLocal(valueVn));
    }

    /**
     * A text read yields strings: {@code f.read()} and {@code f.readline()} on the object {@code
     * open} returns yield a string, and {@code f.readlines()} on it, or {@code s.splitlines()} and
     * {@code s.split(...)} on a string, yield a list of strings. The receiver is checked when its
     * points-to set arrives, so a method of the same name on another object is untouched. Without
     * this a text dataset built from a file's lines carried no element type at all.
     *
     * @param inst The call.
     */
    private void processTextRead(PythonInvokeInstruction inst) {
      SSAInstruction calleeDef = du.getDef(inst.getUse(0));
      if (!(calleeDef instanceof AstPropertyRead)) return;
      AstPropertyRead read = (AstPropertyRead) calleeDef;
      SymbolTable symtab = ir.getSymbolTable();
      if (!symtab.isConstant(read.getMemberRef())) return;
      Object member = symtab.getConstantValue(read.getMemberRef());
      TextReadOperator.Kind kind;
      if ("read".equals(member) || "readline".equals(member))
        kind = TextReadOperator.Kind.FILE_TEXT;
      else if ("readlines".equals(member)) kind = TextReadOperator.Kind.FILE_LINES;
      else if ("splitlines".equals(member) || "split".equals(member))
        kind = TextReadOperator.Kind.STRING_PIECES;
      else return;
      TextReadOperator operator =
          getBuilder()
          .new TextReadOperator(node, inst.iIndex(), getPointerKeyForLocal(inst.getDef()), kind);
      int receiverVn = read.getObjectRef();
      PointerKey receiverKey = getPointerKeyForLocal(receiverVn);
      // A literal receiver (`"a,b".split(",")`) has an implicitly represented key, which a side
      // effect must not touch (wala/ML#668): read its contents here and apply the operator once.
      if (contentsAreInvariant(symtab, du, receiverVn) || system.isImplicit(receiverKey)) {
        operator.apply(getInvariantContents(symtab, du, node, receiverVn));
        return;
      }
      system.newSideEffect(operator, receiverKey);
    }

    /**
     * The dictionary methods a configuration round trip goes through, read and written as the
     * fields a dictionary's constant keys name (<a
     * href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): {@code d.pop(key, default)}
     * and {@code d.get(key, default)} yield the field a constant {@code key} names, and the default
     * when one is given; {@code d.items()} yields a list of one {@code (key, value)} tuple whose
     * fields are the dictionary's catalogued keys and their fields' values, so {@code for k, v in
     * d.items()} binds {@code v} to the values; and {@code d.update(other)} writes each of {@code
     * other}'s catalogued fields to the field of the same name of {@code d}, and {@code other}'s
     * keys into {@code d}'s catalog, so a later {@code **d} or {@code d.items()} sees them. The
     * receiver is checked when its points-to set arrives, so a method of the same name on an object
     * that is not a dictionary is untouched, and a receiver that is implicitly represented (a
     * literal) is read directly, since a side effect must not materialize its key (wala/ML#668).
     *
     * <p>Without this, a model's {@code from_config} that deep-copies its configuration, pops the
     * layer configurations off it, rebuilds each layer in a loop over their items and binds the
     * rebuilt layers through {@code update} and {@code **} received no layer at all, and the
     * rebuilt model's forward pass was empty.
     *
     * @param inst The call.
     */
    private void processDictMethod(PythonInvokeInstruction inst) {
      SSAInstruction calleeDef = du.getDef(inst.getUse(0));
      if (!(calleeDef instanceof AstPropertyRead read)) return;
      SymbolTable symtab = ir.getSymbolTable();
      if (!symtab.isConstant(read.getMemberRef())) return;
      Object member = symtab.getConstantValue(read.getMemberRef());
      DictMethodOperator.Kind kind;
      int positional = inst.getNumberOfPositionalParameters();
      String key = null;
      int argument = -1;
      if (("pop".equals(member) || "get".equals(member)) && positional >= 2 && positional <= 3) {
        int keyVn = inst.getUse(1);
        if (!symtab.isConstant(keyVn) || !(symtab.getConstantValue(keyVn) instanceof String name))
          return;
        key = name;
        kind = DictMethodOperator.Kind.FIELD;
        if (positional == 3) argument = inst.getUse(2);
      } else if ("items".equals(member) && positional == 1) kind = DictMethodOperator.Kind.ITEMS;
      else if ("update".equals(member) && positional == 2) {
        kind = DictMethodOperator.Kind.UPDATE;
        argument = inst.getUse(1);
      } else return;
      if (kind != DictMethodOperator.Kind.UPDATE && !inst.hasDef()) return;
      PointerKey resultKey = inst.hasDef() ? getPointerKeyForLocal(inst.getDef()) : null;
      // The default of a read, or the argument of an update: its keys when it is invariant, its
      // pointer key otherwise.
      InstanceKey[] argumentKeys = null;
      PointerKey argumentKey = null;
      if (argument >= 0) {
        if (contentsAreInvariant(symtab, du, argument))
          argumentKeys = getInvariantContents(symtab, du, node, argument);
        else argumentKey = getPointerKeyForLocal(argument);
      }
      DictMethodOperator operator =
          getBuilder()
          .new DictMethodOperator(
              node, inst.iIndex(), kind, key, resultKey, argumentKeys, argumentKey);
      int receiverVn = read.getObjectRef();
      PointerKey receiverKey = getPointerKeyForLocal(receiverVn);
      if (contentsAreInvariant(symtab, du, receiverVn) || system.isImplicit(receiverKey)) {
        operator.apply(getInvariantContents(symtab, du, node, receiverVn));
        return;
      }
      system.newSideEffect(operator, receiverKey);
    }

    /**
     * Binop allocation synthesis is intentionally a no-op. An earlier version of this method
     * registered a per-instruction {@link NewSiteReference} (keyed by the SSA instruction index) so
     * the pointer analysis could track binop results (wala/ML#398). That fixed {@code
     * testBinopThroughDataset} but caused a regression elsewhere: seeding the binop def's PTS with
     * a non-tensor-producing {@code Lobject} instance key suppressed tensor identification for
     * downstream values, dropping counts on {@code testNeuralNetwork}, {@code testAutoencoder*},
     * and similar. Preserving tensor identification is a merge-safety invariant against {@code
     * master}, so the allocation is disabled pending a narrower approach. The {@link
     * PythonBinaryOpInstruction} scaffolding and factory override remain in place so a future
     * gated-allocation strategy can hook here. See wala/ML#398.
     */
    @Override
    public void visitPythonBinaryOp(PythonBinaryOpInstruction binop) {
      processArrayOperation(binop);
      // List repetition and concatenation (wala/ML#960): `xs * n` and `xs + ys` produce a fresh
      // list (or tuple) whose elements are the operands' elements. Only a list or tuple key
      // flowing into an operand produces anything, so a tensor binop keeps its empty result set
      // and tensor identification is untouched (the wala/ML#398 regression class). The fresh key
      // is allocated at this instruction's index, so `xs + ys` over two lists yields ONE fresh
      // list, and the elements go to the synthetic {@value #LIST_OPERATION_CONTENTS_FIELD} field
      // that every non-constant subscript, iteration and `zip` read beside the append contents:
      // indices are unknowable here, a numeric field would let a length-counting reader derive a
      // wrong extent, and the append field would let the shape readers that interpret appended
      // contents derive one.
      IBinaryOpInstruction.IOperator operator = binop.getOperator();
      if (operator != IBinaryOpInstruction.Operator.ADD
          && operator != IBinaryOpInstruction.Operator.MUL) return;
      PointerKey resultKey = getPointerKeyForLocal(binop.getDef());
      SymbolTable symtab = ir.getSymbolTable();
      int[] operands = {binop.getUse(0), binop.getUse(1)};
      if (operator == IBinaryOpInstruction.Operator.ADD) {
        int prefix = literalTupleLength(symtab, du, operands[0]);
        if (prefix > 0)
          getBuilder().exactListOperationPrefixes.put(Pair.make(node, binop.iIndex()), prefix);
      }
      PointerKey[] keys = new PointerKey[2];
      InstanceKey[][] invariant = new InstanceKey[2][];
      for (int i = 0; i < 2; i++) {
        int use = operands[i];
        if (use <= 0 || symtab.isConstant(use)) continue; // the multiplier, or a literal
        keys[i] = getPointerKeyForLocal(use);
        // An operand whose contents are invariant (a literal list) or whose key is otherwise
        // represented implicitly (a summary return) is read directly, as the append model does:
        // a constraint over such a key would crash `findOrCreatePointsToSet` (the wala/ML#668
        // trap), and even a side effect would MATERIALIZE the key, turning an implicit parameter
        // or return explicit and changing how the tensor analysis reads it (a parameter with no
        // list evidence at all gained a tensor state that way).
        if (contentsAreInvariant(symtab, du, use) || system.isImplicit(keys[i]))
          invariant[i] = getInvariantContents(symtab, du, node, use);
      }
      for (int i = 0; i < 2; i++) {
        if (keys[i] == null) continue;
        int other = 1 - i;
        ListOperationOperator listOperation =
            getBuilder()
            .new ListOperationOperator(
                node, binop.iIndex(), resultKey, operator, i, keys[other], invariant[other]);
        if (invariant[i] != null) {
          for (InstanceKey key : invariant[i]) listOperation.contribute(key);
        } else {
          // A side effect, not an assignment constraint: the flow graph the tensor dataflow walks
          // includes every unary constraint's edge, and an operand-to-result edge here would carry
          // a tensor operand's state into the result of `x * y` (the wala/ML#405 substrate-leak
          // class). The side effect adds only the fresh key.
          system.newSideEffect(listOperation, keys[i]);
        }
        // A key declined because the other operand has not grown into the rule yet is retried
        // when that operand's set changes, so the operator also watches it (a represented one;
        // an invariant or implicit other operand has static contents and nothing arrives late).
        if (keys[other] != null && invariant[other] == null)
          system.newSideEffect(listOperation, keys[other]);
      }
    }

    /**
     * Allocates the result of a binary operator over an array (wala/ML#1009): for each operand key
     * of a type {@link #freshBinaryOpResultTypes} names, a fresh key of the mapped type at the
     * operator's instruction index joins the result. The fresh key is added by an instance
     * constraint, not an assignment edge, so no operand's tensor state flows into the result
     * through the flow graph (the wala/ML#405 substrate-leak class); the result's type comes from
     * the operator's own generator.
     *
     * @param binop The binary operator.
     */
    private void processArrayOperation(PythonBinaryOpInstruction binop) {
      Map<TypeReference, TypeReference> types = getBuilder().freshBinaryOpResultTypes;
      if (types.isEmpty() || !binop.hasDef()) return;
      PointerKey resultKey = getPointerKeyForLocal(binop.getDef());
      SymbolTable symtab = ir.getSymbolTable();
      ArrayOperationOperator operator =
          getBuilder().new ArrayOperationOperator(node, binop.iIndex(), resultKey, types);
      for (int i = 0; i < 2; i++) {
        int use = binop.getUse(i);
        if (use <= 0 || symtab.isConstant(use)) continue;
        PointerKey key = getPointerKeyForLocal(use);
        // As for list operations: an invariant or implicit operand is read directly, since a
        // constraint over its key would materialize it (the wala/ML#668 trap).
        if (contentsAreInvariant(symtab, du, use) || system.isImplicit(key))
          for (InstanceKey ik : getInvariantContents(symtab, du, node, use))
            operator.contribute(ik);
        else system.newSideEffect(operator, key);
      }
    }

    /**
     * Allocates the element read off an array (wala/ML#1009): for each receiver key of a type
     * {@link #freshElementTypes} names, a fresh key of the mapped type at the read's instruction
     * index joins the result, with the receiver's per-instance methods, so a slice of the element
     * and arithmetic over it allocate as they do over the receiver. Without a key, every read
     * downstream of the element was empty, and the operators over it read nothing. The element's
     * shape and dtype are the read's own generator's, which peels the receiver's leading axis. The
     * read is an element read when its member is drawn by iteration (a loop variable, or the
     * element a summary reads for one) or is an index: a constant integer, or a value that is not a
     * literal tuple (the ellipsis and newaxis forms, which add axes and keep their own modeling). A
     * string member is an attribute and {@code None} alone adds an axis; neither is an element.
     *
     * @param read The property read.
     */
    private void processElementRead(AstPropertyRead read) {
      Map<TypeReference, TypeReference> types = getBuilder().freshElementTypes;
      if (types.isEmpty() || !read.hasDef()) return;
      SymbolTable symtab = ir.getSymbolTable();
      int member = read.getMemberRef();
      if (!isLoopVariable(read.getDef())) {
        if (symtab.isConstant(member)) {
          // An integer literal is a `Long` constant in this front end; a string is an attribute.
          Object index = symtab.getConstantValue(member);
          if (!(index instanceof Long) && !(index instanceof Integer)) return;
        } else if (du.getDef(member) instanceof SSANewInstruction) return;
      }
      int object = read.getObjectRef();
      if (object <= 0 || symtab.isConstant(object)) return;
      PointerKey resultKey = getPointerKeyForLocal(read.getDef());
      ArrayOperationOperator operator =
          getBuilder().new ArrayOperationOperator(node, read.iIndex(), resultKey, types);
      PointerKey key = getPointerKeyForLocal(object);
      if (contentsAreInvariant(symtab, du, object) || system.isImplicit(key))
        for (InstanceKey ik : getInvariantContents(symtab, du, node, object))
          operator.contribute(ik);
      else system.newSideEffect(operator, key);
    }

    @Override
    public void visitArrayLoad(SSAArrayLoadInstruction inst) {
      newFieldRead(node, inst.getArrayRef(), inst.getIndex(), inst.getDef());
    }

    @Override
    public void visitArrayStore(SSAArrayStoreInstruction inst) {
      newFieldWrite(node, inst.getArrayRef(), inst.getIndex(), inst.getValue());
    }

    @Override
    public void visitPropertyWrite(AstPropertyWrite instruction) {
      if (!processStarredElement(instruction)) super.visitPropertyWrite(instruction);
    }

    /**
     * Unpacks a starred element of a tuple or list literal, {@code (a, *rest)} (wala/ML#989). The
     * parser writes such a literal's elements to its order-free operation-contents property, and a
     * starred one under {@link PythonCAstToIRTranslator#STARRED_ARGUMENT_MARKER}; that write stands
     * for every element of the iterable, not the iterable itself, so the iterable's elements flow
     * into the literal's operation contents instead of the iterable being stored as one element.
     *
     * @param instruction The property write.
     * @return {@code true} iff the write was a starred element and has been handled.
     */
    private boolean processStarredElement(AstPropertyWrite instruction) {
      SymbolTable symtab = ir.getSymbolTable();
      int memberRef = instruction.getMemberRef();
      if (!symtab.isConstant(memberRef)
          || !PythonCAstToIRTranslator.STARRED_ARGUMENT_MARKER.equals(
              symtab.getConstantValue(memberRef))) return false;
      int objectVn = instruction.getObjectRef();
      // A guard, not a path: the parser writes the marker only into the literal it allocates in
      // the same body, whose contents are invariant. Were it otherwise, the ordinary write would
      // store the iterable under the marker name, which no reader consults.
      if (!contentsAreInvariant(symtab, du, objectVn)) return false;
      IField contents = resolveRootField(getClassHierarchy(), LIST_OPERATION_CONTENTS_FIELD);
      if (contents == null) return false;
      int valueVn = instruction.getValue();
      PointerKey valueKey = getPointerKeyForLocal(valueVn);
      for (InstanceKey literal : getInvariantContents(symtab, du, node, objectVn)) {
        PointerKey target =
            ((AstPointerKeyFactory) getBuilder().getPointerKeyFactory())
                .getPointerKeyForInstanceField(literal, contents);
        StarredElementOperator operator =
            getBuilder().new StarredElementOperator(target, valueKey, instruction.iIndex());
        if (contentsAreInvariant(symtab, du, valueVn) || system.isImplicit(valueKey))
          for (InstanceKey iterable : getInvariantContents(symtab, du, node, valueVn))
            operator.read(iterable);
        else system.newSideEffect(operator, valueKey);
      }
      return true;
    }

    @Override
    public void visitPropertyRead(AstPropertyRead instruction) {
      // An element read over an object whose class declares the iteration protocol reads nothing:
      // such an object's elements come from `__next__`, and its properties are its attributes,
      // not its elements (wala/ML#1010).
      elementRead = isLoopVariable(instruction.getDef());
      try {
        super.visitPropertyRead(instruction);
      } finally {
        elementRead = false;
      }
      processListContentsRead(instruction);
      processNegativeSubscript(instruction);
      processUnknownIndexRead(instruction);
      processElementRead(instruction);

      if (this.ir.getSymbolTable().isConstant(instruction.getMemberRef())) {
        Object constantValue =
            this.ir.getSymbolTable().getConstantValue(instruction.getMemberRef());

        if (Objects.equals(constantValue, IMPORT_WILDCARD_CHARACTER)) {
          // We have a wildcard.
          logger.fine(
              "Detected wildcard for " + instruction.getMemberRef() + " in " + instruction + ".");

          processWildcardImports(instruction);
        }

        // check if we are reading from an module initialization script.
        SSAInstruction objRefDef = du.getDef(instruction.getObjectRef());
        logger.finest(
            () ->
                "Found def: "
                    + objRefDef
                    + " for object reference: "
                    + instruction.getObjectRef()
                    + " in instruction: "
                    + instruction
                    + ".");

        if (objRefDef instanceof AstGlobalRead) {
          AstGlobalRead agr = (AstGlobalRead) objRefDef;
          String fieldName = getStrippedDeclaredFieldName(agr);
          logger.finer("Found field name: " + fieldName);

          // if the "receiver" is a module initialization script.
          if (fieldName.endsWith("/" + MODULE_INITIALIZATION_FILENAME))
            try {
              processWildcardImports(instruction, fieldName, constantValue.toString());
            } catch (CancelException e) {
              throw new CancelRuntimeException(e);
            }
        }
      }
    }

    /**
     * Processes the given {@link AstPropertyRead} for any potential wildcard imports being utilized
     * by the instruction.
     *
     * @param instruction The {@link AstPropertyRead} whose definition may depend on a wildcard
     *     import.
     */
    private void processWildcardImports(AstPropertyRead instruction) {
      int objRef = instruction.getObjectRef();
      logger.fine("Seeing if " + objRef + " refers to an import.");

      SSAInstruction def = this.du.getDef(objRef);
      logger.finer("Found definition: " + def + ".");

      TypeName scriptTypeName = this.ir.getMethod().getReference().getDeclaringClass().getName();
      logger.finer("Found script: " + scriptTypeName + ".");

      String scriptName = getScriptName(scriptTypeName);
      logger.fine("Script name is: " + scriptName);
      assert scriptName.endsWith("." + PYTHON_FILE_EXTENSION);

      if (def instanceof SSAInvokeInstruction) {
        // Library case.
        SSAInvokeInstruction invokeInstruction = (SSAInvokeInstruction) def;
        MethodReference declaredTarget = invokeInstruction.getDeclaredTarget();
        Atom declaredTargetName = declaredTarget.getName();

        if (declaredTargetName.equals(IMPORT_FUNCTION_NAME)) {
          // It's an import "statement" importing a library.
          logger.fine("Found library import statement in: " + scriptTypeName + ".");

          logger.info(
              "Adding: "
                  + declaredTarget.getDeclaringClass().getName().toString().substring(1)
                  + " to wildcard imports for: "
                  + scriptName
                  + ".");

          // Add the library to the script's queue of wildcard imports.
          getBuilder()
              .getScriptToWildcardImports()
              .compute(
                  scriptName,
                  (_, v) -> {
                    if (v == null) {
                      Deque<MethodReference> deque = new ArrayDeque<>();
                      deque.push(declaredTarget);
                      return deque;
                    } else {
                      v.push(declaredTarget);
                      return v;
                    }
                  });
        }
      } else if (def instanceof SSAGetInstruction) {
        // We are importing from a script.
        SSAGetInstruction getInstruction = (SSAGetInstruction) def;
        String strippedFieldName = getStrippedDeclaredFieldName(getInstruction);

        MethodReference methodReference = getMethodReferenceRepresentingScript(strippedFieldName);

        logger.info(
            "Adding: "
                + methodReference.getDeclaringClass().getName().toString().substring(1)
                + " to wildcard imports for: "
                + scriptName
                + ".");

        // Add the script to the queue of this script's wildcard imports.
        getBuilder()
            .getScriptToWildcardImports()
            .compute(
                scriptName,
                (_, v) -> {
                  if (v == null) {
                    Deque<MethodReference> deque = new ArrayDeque<>();
                    deque.push(methodReference);
                    return deque;
                  } else {
                    v.push(methodReference);
                    return v;
                  }
                });
      } else if (def instanceof AstPropertyRead) processWildcardImports((AstPropertyRead) def);
      else
        throw new IllegalArgumentException(
            "Not expecting the definition: "
                + def
                + " of the object reference of: "
                + instruction
                + " to be: "
                + def.getClass());
    }

    /**
     * Given a script's name, returns the {@link MethodReference} representing the script.
     *
     * @param scriptName The name of the script.
     * @return The corresponding {@link MethodReference} representing the script.
     */
    private static MethodReference getMethodReferenceRepresentingScript(String scriptName) {
      TypeReference typeReference =
          TypeReference.findOrCreate(PythonTypes.pythonLoader, "L" + scriptName);

      return MethodReference.findOrCreate(
          typeReference,
          Atom.findOrCreateAsciiAtom(DO_METHOD_NAME),
          Descriptor.findOrCreate(null, PythonTypes.rootTypeName));
    }

    @Override
    public void visitAstGlobalRead(AstGlobalRead globalRead) {
      super.visitAstGlobalRead(globalRead);

      TypeName enclosingMethodTypeName =
          this.ir.getMethod().getReference().getDeclaringClass().getName();

      String scriptName = getScriptName(enclosingMethodTypeName);

      if (scriptName.endsWith("." + PYTHON_FILE_EXTENSION)) {
        // We have a valid script name.
        logger.fine("Script name is: " + scriptName);
        String fieldName = getStrippedDeclaredFieldName(globalRead);
        try {
          processWildcardImports(globalRead, scriptName, fieldName);
        } catch (CancelException e) {
          throw new CancelRuntimeException(e);
        }
      }
    }

    /**
     * Returns the name of the script for the given {@link TypeName} representing a the name of a
     * method.
     *
     * @param methodName The name of the method.
     * @return The name of the corresponding script.
     * @implNote In Ariadne, scripts are also "methods" with the name "do."
     */
    private static String getScriptName(TypeName methodName) {
      String composed =
          methodName.getPackage() == null
              ? methodName.getClassName().toString()
              : methodName.getPackage().toString() + "/" + methodName.getClassName().toString();

      // A method nested in a class (e.g. `script layers/feed_forward.py/Conv1d/call`) composes to
      // a name whose script segment is interior, not terminal. Truncate at the script's file
      // extension so reads inside class methods key the same script as module-level reads;
      // without this, the wildcard lookup is skipped for them (wala/ML#665).
      String marker = "." + PYTHON_FILE_EXTENSION + "/";
      int extension = composed.indexOf(marker);
      if (extension >= 0) {
        return composed.substring(0, extension + marker.length() - 1);
      }

      if (composed.endsWith("." + PYTHON_FILE_EXTENSION)) {
        return composed;
      }

      return (methodName.getPackage() == null ? methodName.getClassName() : methodName.getPackage())
          .toString();
    }

    /**
     * Processes the given {@link SSAInstruction} for any potential wildcard imports being utilized
     * by the instruction.
     *
     * @param instruction The {@link SSAInstruction} whose definition may depend on a wildcard
     *     import.
     * @param scriptName The name of the script to check for wildcard imports.
     * @param fieldName The name of the field that may be imported using a wildcard.
     */
    private void processWildcardImports(
        SSAInstruction instruction, String scriptName, String fieldName) throws CancelException {
      // Get the method reference for the given script.
      MethodReference reference = getMethodReferenceRepresentingScript(scriptName);

      // Get the nodes for the script.
      Set<CGNode> scriptNodes = this.getBuilder().getCallGraph().getNodes(reference);

      // For each node representing the script.
      for (CGNode node : scriptNodes) {
        // if we haven't visited the node yet.
        if (!this.getBuilder().haveAlreadyVisited(node)) {
          // visit the node first. Otherwise, we won't know if there are any wildcard imports in
          // it.
          this.getBuilder().addConstraintsFromNode(node, null);

          assert this.getBuilder().haveAlreadyVisited(node);
        }
      }

      // Are there any wildcard imports for this script?
      if (getBuilder().getScriptToWildcardImports().containsKey(scriptName)) {
        logger.info("Found wildcard imports in " + scriptName + " for " + instruction + ".");

        Deque<MethodReference> deque = getBuilder().getScriptToWildcardImports().get(scriptName);

        for (MethodReference importMethodReference : deque) {
          logger.fine(
              "Library with wildcard import is: "
                  + importMethodReference.getDeclaringClass().getName().toString().substring(1)
                  + ".");

          logger.fine("Examining global: " + fieldName + " for wildcard import.");

          CallGraph callGraph = this.getBuilder().getCallGraph();
          Set<CGNode> nodes = callGraph.getNodes(importMethodReference);

          if (nodes.isEmpty())
            logger.warning(
                "Can't find CG node for import method: "
                    + importMethodReference.getSignature()
                    + ".");

          PointerKey defPK = this.getPointerKeyForLocal(instruction.getDef());
          assert defPK != null;

          for (CGNode n : nodes) {
            for (Iterator<NewSiteReference> nit = n.iterateNewSites(); nit.hasNext(); ) {
              NewSiteReference newSiteReference = nit.next();

              String name = newSiteReference.getDeclaredType().getName().getClassName().toString();
              logger.finest("Examining: " + name + ".");

              if (name.equals(fieldName)) {
                logger.info("Found wildcard import for: " + name + ".");

                InstanceKey instanceKey =
                    this.getBuilder().getInstanceKeyForAllocation(n, newSiteReference);

                if (this.system.newConstraint(defPK, instanceKey)) {
                  logger.fine("Added constraint that: " + defPK + " gets: " + instanceKey + ".");
                  return;
                }
              }
            }

            // Also check the put instructions, as these may be generated by the initialization
            // file.
            n.getIR()
                .visitNormalInstructions(
                    new PythonInstructionVisitor() {

                      @Override
                      public void visitPut(SSAPutInstruction putInstruction) {
                        FieldReference putField = putInstruction.getDeclaredField();

                        if (fieldName.equals(putField.getName().toString())) {
                          // Found it.
                          int putVal = putInstruction.getVal();

                          // Make the def point to the put instruction value.
                          PointerKey putValPK = getBuilder().getPointerKeyForLocal(n, putVal);

                          if (system.newConstraint(defPK, assignOperator, putValPK))
                            logger.fine(
                                "Added constraint that: " + defPK + " gets: " + putValPK + ".");
                        }
                      }
                    });

            // Also check the module's named bindings (wala/ML#665). Python's wildcard exports
            // every public module-level name, including modules the source module itself
            // imported (`import tensorflow as tf` makes `tf` an exported binding), and such a
            // binding is neither an allocation named after the field (the `def`/`class` case
            // above) nor a `put` onto a module object: it is a named local of the script body
            // holding the import's result. Match the script's local names and assign the
            // binding's value to the reader.
            if (n.getMethod() instanceof AstMethod) {
              for (Iterator<SSAInstruction> it = n.getIR().iterateAllInstructions();
                  it.hasNext(); ) {
                SSAInstruction inst = it.next();
                if (!inst.hasDef() || inst.iIndex() < 0) continue;

                String[] localNames = n.getIR().getLocalNames(inst.iIndex(), inst.getDef());
                if (localNames == null) continue;

                for (String localName : localNames) {
                  if (fieldName.equals(localName)) {
                    PointerKey boundValuePK = getBuilder().getPointerKeyForLocal(n, inst.getDef());

                    if (system.newConstraint(defPK, assignOperator, boundValuePK))
                      logger.fine(
                          "Added wildcard binding constraint that: "
                              + defPK
                              + " gets: "
                              + boundValuePK
                              + " (wala/ML#665).");
                  }
                }
              }
            }
          }
        }
      }
    }

    private static String getStrippedDeclaredFieldName(SSAGetInstruction instruction) {
      String declaredFieldName = instruction.getDeclaredField().getName().toString();
      assert declaredFieldName.startsWith(GLOBAL_IDENTIFIER + " ");

      // Remove the global identifier.
      return declaredFieldName.substring(
          (GLOBAL_IDENTIFIER + " ").length(), declaredFieldName.length());
    }
  }

  @Override
  protected void processCallingConstraints(
      CGNode caller,
      SSAAbstractInvokeInstruction instruction,
      CGNode target,
      InstanceKey[][] constParams,
      PointerKey uniqueCatchKey) {

    if (!(instruction instanceof PythonInvokeInstruction)) {
      super.processCallingConstraints(caller, instruction, target, constParams, uniqueCatchKey);
    } else {
      MutableIntSet args = IntSetUtil.make();

      // positional parameters
      PythonInvokeInstruction call = (PythonInvokeInstruction) instruction;
      StarArguments star = new StarArguments(caller, call, target, constParams);
      for (int i = 0; i < call.getNumberOfPositionalParameters(); i++) {
        if (star.bindPositional(i)) continue;
        if (i >= target.getMethod().getNumberOfParameters()) continue;
        PointerKey lval = getPointerKeyForLocal(target, i + 1);
        args.add(i);

        if (constParams != null && constParams[i] != null) {
          InstanceKey[] ik = constParams[i];
          for (InstanceKey element : ik) {
            system.newConstraint(lval, element);
          }
        } else {
          PointerKey rval = getPointerKeyForLocal(caller, call.getUse(i));

          // If we are looking at the implicit parameter of a callable.
          if (call.getCallSite().isDispatch()
              && isCallable(target.getMethod().getReference())
              && i == 0
              && refersToAnObject(rval)) {
            // Ensure that lval's variable refers to the callable method instead of callable object,
            // precisely linking the object to its trampoline using the `__call__` (or similar)
            // field. When the target is a trampoline keyed on a receiver instance (wala/ML#679),
            // link only that receiver's field; the site's other receivers dispatch to their own
            // trampoline nodes. Non-trampoline targets may inherit a receiver-keyed context whose
            // instance is unrelated to this site's receivers, so they are not restricted.
            ContextItem receiverItem =
                target.getMethod().getDeclaringClass() instanceof PythonInstanceMethodTrampoline
                    ? target.getContext().get(ContextKey.RECEIVER)
                    : null;
            InstanceKey contextReceiver =
                receiverItem instanceof InstanceKey ? (InstanceKey) receiverItem : null;

            IClassHierarchy cha = getClassHierarchy();

            Atom[] possibleFields = {
              Atom.findOrCreateUnicodeAtom(CALLABLE_METHOD_NAME),
              Atom.findOrCreateUnicodeAtom(CALLABLE_METHOD_NAME_FOR_KERAS_MODELS),
              Atom.findOrCreateUnicodeAtom(DO_METHOD_NAME)
            };

            getSystem()
                .newSideEffect(
                    new AbstractOperator<PointsToSetVariable>() {
                      @Override
                      public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable[] rhs) {
                        if (rhs[0].getValue() != null) {
                          rhs[0]
                              .getValue()
                              .foreach(
                                  i -> {
                                    InstanceKey ik = getSystem().getInstanceKey(i);
                                    if (contextReceiver != null && !contextReceiver.equals(ik))
                                      return;

                                    for (Atom fieldName : possibleFields) {
                                      FieldReference fieldRef =
                                          FieldReference.findOrCreate(
                                              PythonTypes.Root, fieldName, PythonTypes.Root);

                                      IField f = cha.resolveField(fieldRef);

                                      if (f != null) {
                                        PointerKey fieldPK =
                                            getPointerKeyFactory()
                                                .getPointerKeyForInstanceField(ik, f);

                                        getSystem().newConstraint(lval, assignOperator, fieldPK);
                                      }
                                    }
                                  });
                        }
                        return NOT_CHANGED;
                      }

                      @Override
                      public int hashCode() {
                        return lval.hashCode() ^ rval.hashCode();
                      }

                      @Override
                      public boolean equals(Object o) {
                        return this == o;
                      }

                      @Override
                      public String toString() {
                        return "precise trampoline link for " + rval;
                      }
                    },
                    new PointerKey[] {rval});
          } else {
            // A receiver-keyed target (wala/ML#679) filters the dispatched parameter to the
            // context's receiver instance; every other target binds the full argument set.
            PointerKey formal = i == 0 ? getReceiverFilteredPointerKey(target, lval) : lval;
            getSystem()
                .newConstraint(
                    formal,
                    formal instanceof FilteredPointerKey ? filterOperator : assignOperator,
                    rval);
          }
        }
      }

      // keyword arguments
      int paramNumber = call.getNumberOfPositionalParameters();
      keywords:
      for (String argName : call.getKeywords()) {
        int src = call.getUse(argName);
        if (star.bindDoubleStarred(
            argName,
            src,
            constParams != null && paramNumber < constParams.length
                ? constParams[paramNumber]
                : null)) {
          paramNumber++;
          continue;
        }
        for (int i = 0; i < target.getIR().getSymbolTable().getMaxValueNumber(); i++) {
          String[] paramNames = target.getIR().getLocalNames(0, i + 1);
          if (paramNames != null) {
            for (String destName : paramNames) {
              if (argName.equals(destName)) {
                PointerKey lval = getPointerKeyForLocal(target, i + 1);
                args.add(i);
                int p = paramNumber;
                if (constParams != null && constParams[p] != null) {
                  InstanceKey[] ik = constParams[p];
                  for (InstanceKey element : ik) {
                    system.newConstraint(lval, element);
                  }
                } else {
                  PointerKey rval = getPointerKeyForLocal(caller, src);
                  getSystem().newConstraint(lval, assignOperator, rval);
                }
                paramNumber++;
                continue keywords;
              }
            }
          }
        }
        // no such argument in callee: a `**kwargs` formal collects it (wala/ML#991)
        star.packKeyword(
            argName,
            constParams != null && paramNumber < constParams.length
                ? constParams[paramNumber]
                : null,
            src);
        paramNumber++;
      }

      // A synthesized constructor's summary declares the wrapped `__init__`'s parameter count,
      // although its own formals stop one short (formal i mirrors `__init__`'s parameter i + 1),
      // so its defaulted range shifts down by one (wala/ML#762).
      boolean ctorTarget = target.getMethod() instanceof PythonConstructorFunction;
      int numParams = target.getMethod().getNumberOfParameters() - (ctorTarget ? 1 : 0);
      // The `*args`, `**kwargs`, and keyword-only formals are parameters but cannot receive a
      // positional default, and the parser appends them after the plain positionals. Counting the
      // defaulted range back from the end of the formal list therefore has to discount them, or
      // every default binds one parameter to the right and the first defaulted parameter binds
      // nothing (wala/ML#843). The writer in `PythonCAstToIRTranslator` discounts them the same
      // way, so the two sides agree on which index carries which default.
      int trailing =
          target.getMethod() instanceof StarFormalDeclaration
              ? ((StarFormalDeclaration) target.getMethod())
                  .getNumberOfTrailingNonDefaultableParameters()
              : 0;
      int last = numParams - trailing;
      int dflts = last - target.getMethod().getNumberOfDefaultParameters();
      for (int i = dflts; i < last; i++) {
        if (!args.contains(i)) {
          // A synthesized constructor's trailing formal i mirrors `__init__`'s parameter i + 1,
          // and the default globals are written under `__init__`'s entity name, so the lookup
          // follows that mapping (wala/ML#762). Every other target reads its own entity's global.
          String name = defaultsGlobalName(target, i, "_defaults_");
          IField f = resolveGlobal(name);
          // Built lazily: a node's context can share structure deeply enough that rendering it
          // overflows a string, and this runs for every unbound default whatever the log level.
          int param = i;
          logger.fine(
              () ->
                  "DEFAULTS-BIND target "
                      + target
                      + " param "
                      + param
                      + " global "
                      + name
                      + " field "
                      + f);
          PointerKey lval = getPointerKeyForLocal(target, i + 1);
          getSystem().newConstraint(lval, assignOperator, new StaticFieldKey(f));
          // A `@click.option` default is written under a global of its own and binds under a
          // constant key of its own class (wala/ML#971); the global of a parameter with no click
          // option holds nothing and contributes nothing. Globals are dynamic fields of `Root`, so
          // the name always resolves.
          String clickName = defaultsGlobalName(target, i, "_click_defaults_");
          getSystem()
              .newConstraint(
                  lval, new ClickDefaultOperator(), new StaticFieldKey(resolveGlobal(clickName)));
        }
      }

      // return values
      PointerKey rret = getPointerKeyForReturnValue(target);
      PointerKey lret = getPointerKeyForLocal(caller, call.getReturnValue(0));
      getSystem().newConstraint(lret, assignOperator, rret);

      PointerKey reret = getPointerKeyForExceptionalReturnValue(target);
      PointerKey leret = getPointerKeyForLocal(caller, call.getException());
      getSystem().newConstraint(leret, assignOperator, reret);

      if (target.getMethod().getDeclaringClass().getReference().equals(PythonTypes.SLICE_BUILTIN))
        processSliceResult(caller, call, constParams);
    }
  }

  /**
   * Supplies the result of a {@code slice} builtin call (wala/ML#916). The builtin's body returns
   * nothing; the result is the first argument's points-to set, as the body used to return, except
   * that a key whose concrete type is one of the {@link #freshSliceResultTypes} becomes a fresh
   * allocation of the type it maps to at this call, so a tensor's slice is a tensor of its own
   * rather than an alias of its receiver. The result is a unary constraint from the receiver to the
   * result, an edge of the assignment graph like the one the body used to make through its
   * parameter and return, because the tensor dataflow analysis uses that graph as its flow graph: a
   * side effect would supply the same keys but sever the edge, and every value flowing through a
   * slice of a pass-through receiver (an array's dtype state, a named tuple's element types) would
   * stop at the call. A constant first argument (the {@code slice(None, n, None)} form a
   * subscript's bounds lower to) flows as the constant, as before.
   *
   * @param caller The node containing the call.
   * @param call The {@code slice} call.
   * @param constParams The call's constant arguments, indexed by positional argument, or {@code
   *     null}.
   */
  private void processSliceResult(
      CGNode caller, PythonInvokeInstruction call, InstanceKey[][] constParams) {
    if (call.getNumberOfPositionalParameters() < 2 || !call.hasDef()) return;
    PointerKey def = getPointerKeyForLocal(caller, call.getDef());
    if (constParams != null && constParams.length > 1 && constParams[1] != null) {
      for (InstanceKey element : constParams[1]) system.newConstraint(def, element);
      return;
    }
    PointerKey receiver = getPointerKeyForLocal(caller, call.getUse(1));
    getSystem().newConstraint(def, new SliceResultOperator(caller, call.iIndex()), receiver);
  }

  /** The most fields of a {@code collections.namedtuple} whose names bind its arguments. */
  private static final int MAX_NAMED_TUPLE_FIELDS = 32;

  /** The type object a {@code collections.namedtuple} call returns. */
  public static final TypeReference NAMED_TUPLE_TYPE =
      TypeReference.findOrCreate(
          PythonTypes.pythonLoader, TypeName.findOrCreate("Lcollections/namedtuple"));

  /** An instance a {@code collections.namedtuple} type constructs. */
  public static final TypeReference NAMED_TUPLE_INSTANCE =
      TypeReference.findOrCreate(
          PythonTypes.pythonLoader, TypeName.findOrCreate("Lcollections/namedtuple/instance"));

  /**
   * Constructs a {@code collections.namedtuple} instance (<a
   * href="https://github.com/wala/ML/issues/1031">wala/ML#1031</a>) at a call whose callee is a
   * namedtuple type: an instance allocated at the call, each positional argument bound to the field
   * the type names at its position and to that position, and each keyword argument to its own field
   * and to its name's position. The type keeps its field names on {@code _fields}, as the {@code
   * namedtuple} summary stores them: a sequence of name strings, or one string of names separated
   * by commas or spaces. Names that have not reached that field yet bind when they arrive.
   * Positions from a starred argument on are not bound, since the elements it spreads are not read
   * here. Equal for the same call, so the propagation system keeps one per call site.
   */
  public final class NamedTupleOperator extends UnaryOperator<PointsToSetVariable> {
    private final CGNode node;
    private final int pc;
    private final PointerKey resultKey;
    private final Object[] positionalArguments;
    private final Map<String, Object> keywordArguments;
    private final Set<InstanceKey> constructed = HashSetFactory.make();

    private NamedTupleOperator(
        CGNode node,
        int pc,
        PointerKey resultKey,
        Object[] positionalArguments,
        Map<String, Object> keywordArguments) {
      this.node = node;
      this.pc = pc;
      this.resultKey = resultKey;
      this.positionalArguments = positionalArguments;
      this.keywordArguments = keywordArguments;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() != null)
        rhs.getValue().foreach(i -> construct(getSystem().getInstanceKey(i)));
      return NOT_CHANGED;
    }

    /**
     * Constructs an instance if a key the callee may be is a namedtuple type.
     *
     * @param type A key the callee may be.
     */
    void construct(InstanceKey type) {
      if (!type.concreteType().getReference().equals(NAMED_TUPLE_TYPE) || !constructed.add(type))
        return;
      InstanceKey instance =
          getInstanceKeyForAllocation(node, TypedSiteReference.at(pc, NAMED_TUPLE_INSTANCE));
      if (instance == null) return;
      getSystem().newConstraint(resultKey, instance);
      IField fields = resolveRootField(getClassHierarchy(), "_fields");
      if (fields == null) return;
      PointerKey fieldsKey = getPointerKeyForInstanceField(type, fields);
      NamesOperator names = new NamesOperator(instance);
      // A keyword argument names its field itself; its position waits for the type's names.
      for (Map.Entry<String, Object> keyword : keywordArguments.entrySet())
        names.bindField(
            instance, resolveRootField(getClassHierarchy(), keyword.getKey()), keyword.getValue());
      getSystem().newSideEffect(names, fieldsKey);
    }

    /** Binds the arguments to an instance as the type's field names arrive. */
    private final class NamesOperator extends UnaryOperator<PointsToSetVariable> {
      private final InstanceKey instance;
      private final Set<InstanceKey> read = HashSetFactory.make();

      private NamesOperator(InstanceKey instance) {
        this.instance = instance;
      }

      @Override
      public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
        if (rhs.getValue() != null)
          rhs.getValue().foreach(i -> readNames(getSystem().getInstanceKey(i)));
        return NOT_CHANGED;
      }

      private void readNames(InstanceKey fieldNames) {
        if (!read.add(fieldNames)) return;
        if (fieldNames instanceof ConstantKey<?> constant
            && constant.getValue() instanceof String text) {
          List<String> names = new ArrayList<>();
          for (String name : text.split("[,\\s]+")) if (!name.isEmpty()) names.add(name);
          bind(names);
          return;
        }
        // A sequence of names: each position's name, which may arrive after this read. A
        // namedtuple names few fields; positions past the bound bind nothing.
        IClassHierarchy cha = getClassHierarchy();
        for (int i = 0; i < MAX_NAMED_TUPLE_FIELDS; i++) {
          IField position = resolveRootField(cha, Integer.toString(i));
          if (position == null) continue;
          PointerKey positionKey = getPointerKeyForInstanceField(fieldNames, position);
          // Names that may be `None`, as a defaulted `names=None` reads, have no fields to read.
          if (positionKey == null) continue;
          final int index = i;
          getSystem()
              .newSideEffect(
                  new UnaryOperator<PointsToSetVariable>() {
                    @Override
                    public byte evaluate(PointsToSetVariable l, PointsToSetVariable r) {
                      if (r.getValue() != null)
                        r.getValue()
                            .foreach(
                                k -> {
                                  if (getSystem().getInstanceKey(k) instanceof ConstantKey<?> name
                                      && name.getValue() instanceof String text)
                                    bindName(index, text);
                                });
                      return NOT_CHANGED;
                    }

                    @Override
                    public int hashCode() {
                      return instance.hashCode() * 31 + index;
                    }

                    @Override
                    public boolean equals(Object o) {
                      return this == o;
                    }

                    @Override
                    public String toString() {
                      return "namedtuple field name " + index + " of " + instance;
                    }
                  },
                  positionKey);
        }
      }

      private void bind(List<String> names) {
        for (int i = 0; i < names.size(); i++) bindName(i, names.get(i));
      }

      /** Binds the argument for the field at a position, named {@code name}. */
      private void bindName(int index, String name) {
        Object argument =
            index < positionalArguments.length
                ? positionalArguments[index]
                : keywordArguments.get(name);
        if (argument == null) return;
        IClassHierarchy cha = getClassHierarchy();
        IField named = resolveRootField(cha, name);
        IField numbered = resolveRootField(cha, Integer.toString(index));
        AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
        getSystem()
            .newConstraint(
                factory.getPointerKeyForObjectCatalog(instance),
                getInstanceKeyForConstant(PythonLanguage.Python.getConstantType(index), index));
        bindField(instance, named, argument);
        bindField(instance, numbered, argument);
      }

      /** Binds an argument's value to a field of the instance. */
      private void bindField(InstanceKey instance, IField field, Object argument) {
        if (field == null || argument == null) return;
        PointerKey target = getPointerKeyForInstanceField(instance, field);
        if (argument instanceof InstanceKey[] keys)
          for (InstanceKey key : keys) getSystem().newConstraint(target, key);
        else getSystem().newConstraint(target, assignOperator, (PointerKey) argument);
      }

      @Override
      public int hashCode() {
        return instance.hashCode();
      }

      @Override
      public boolean equals(Object o) {
        return o instanceof NamesOperator other && other.instance.equals(instance);
      }

      @Override
      public String toString() {
        return "namedtuple field names of " + instance;
      }
    }

    @Override
    public int hashCode() {
      return node.hashCode() * 31 + pc;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof NamedTupleOperator other && other.node.equals(node) && other.pc == pc;
    }

    @Override
    public String toString() {
      return "namedtuple construction at " + node + "@" + pc;
    }
  }

  /**
   * Reads or writes a dictionary through one of its methods as the receiver's keys arrive; see
   * {@code PythonConstraintVisitor#processDictMethod} (wala/ML#997). Equal for the same call, so
   * the propagation system keeps one per call site.
   */
  public final class DictMethodOperator extends UnaryOperator<PointsToSetVariable> {
    public enum Kind {
      FIELD,
      ITEMS,
      UPDATE
    }

    private final CGNode node;
    private final int pc;
    private final Kind kind;
    private final String key;
    private final PointerKey resultKey;
    private final InstanceKey[] argumentKeys;
    private final PointerKey argumentKey;
    private final Set<InstanceKey> read = HashSetFactory.make();
    private boolean argumentFlowed;

    private DictMethodOperator(
        CGNode node,
        int pc,
        Kind kind,
        String key,
        PointerKey resultKey,
        InstanceKey[] argumentKeys,
        PointerKey argumentKey) {
      this.node = node;
      this.pc = pc;
      this.kind = kind;
      this.key = key;
      this.resultKey = resultKey;
      this.argumentKeys = argumentKeys;
      this.argumentKey = argumentKey;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() == null) return NOT_CHANGED;
      java.util.List<InstanceKey> receivers = new java.util.ArrayList<>();
      rhs.getValue().foreach(i -> receivers.add(getSystem().getInstanceKey(i)));
      apply(receivers.toArray(new InstanceKey[0]));
      return NOT_CHANGED;
    }

    /**
     * Applies the method to each dictionary the receiver may be.
     *
     * @param receivers The receiver's instance keys.
     */
    void apply(InstanceKey[] receivers) {
      if (receivers == null) return;
      IClassHierarchy cha = getClassHierarchy();
      IClass dict = cha.lookupClass(PythonTypes.dict);
      if (dict == null) return;
      for (InstanceKey receiver : receivers) {
        if (receiver == null || !cha.isSubclassOf(receiver.concreteType(), dict)) continue;
        if (!read.add(receiver)) continue;
        logger.fine(() -> "dict " + kind + " at " + pc + " in " + node + " on " + receiver);
        switch (kind) {
          case FIELD -> readField(receiver);
          case ITEMS -> items(receiver);
          case UPDATE -> update(receiver);
        }
      }
    }

    /** {@code d.pop(key, default)} and {@code d.get(key, default)}: the field, and the default. */
    private void readField(InstanceKey receiver) {
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IField field = resolveRootField(getClassHierarchy(), key);
      if (field == null) return;
      getSystem()
          .newConstraint(
              resultKey, assignOperator, factory.getPointerKeyForInstanceField(receiver, field));
      if (!argumentFlowed) {
        argumentFlowed = true;
        if (argumentKeys != null) {
          for (InstanceKey k : argumentKeys) if (k != null) getSystem().newConstraint(resultKey, k);
        } else if (argumentKey != null && !getSystem().isImplicit(argumentKey))
          getSystem().newConstraint(resultKey, assignOperator, argumentKey);
      }
    }

    /** {@code d.items()}: a list of one {@code (key, value)} tuple over the catalogued fields. */
    private void items(InstanceKey receiver) {
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IClassHierarchy cha = getClassHierarchy();
      IField zero = resolveRootField(cha, "0");
      IField one = resolveRootField(cha, "1");
      if (zero == null || one == null) return;
      InstanceKey list =
          getInstanceKeyForAllocation(node, TypedSiteReference.at(pc, PythonTypes.list));
      InstanceKey tuple =
          getInstanceKeyForAllocation(node, TypedSiteReference.at(pc, PythonTypes.tuple));
      if (list == null || tuple == null) return;
      InstanceKey zeroKey = getInstanceKeyForConstant(PythonLanguage.Python.getConstantType(0), 0);
      InstanceKey oneKey = getInstanceKeyForConstant(PythonLanguage.Python.getConstantType(1), 1);
      getSystem().newConstraint(resultKey, list);
      getSystem().newConstraint(factory.getPointerKeyForObjectCatalog(list), zeroKey);
      getSystem().newConstraint(factory.getPointerKeyForInstanceField(list, zero), tuple);
      getSystem().newConstraint(factory.getPointerKeyForObjectCatalog(tuple), zeroKey);
      getSystem().newConstraint(factory.getPointerKeyForObjectCatalog(tuple), oneKey);
      // The keys are the receiver's catalogued names; the values are the fields they name.
      getSystem()
          .newConstraint(
              factory.getPointerKeyForInstanceField(tuple, zero),
              assignOperator,
              factory.getPointerKeyForObjectCatalog(receiver));
      getSystem()
          .newSideEffect(
              new ElementCopyOperator(
                  receiver, factory.getPointerKeyForInstanceField(tuple, one), pc),
              factory.getPointerKeyForObjectCatalog(receiver));
    }

    /** {@code d.update(other)}: each of {@code other}'s catalogued fields into {@code d}'s. */
    private void update(InstanceKey receiver) {
      if (argumentKeys != null) for (InstanceKey other : argumentKeys) copyInto(other, receiver);
      else if (argumentKey != null && !getSystem().isImplicit(argumentKey))
        getSystem().newSideEffect(new DictArgumentOperator(receiver, pc), argumentKey);
    }

    /**
     * Flows every catalogued field of {@code other} into the field of the same name of {@code
     * receiver}, and {@code other}'s keys into {@code receiver}'s catalog.
     */
    private void copyInto(InstanceKey other, InstanceKey receiver) {
      if (other == null) return;
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      getSystem()
          .newConstraint(
              factory.getPointerKeyForObjectCatalog(receiver),
              assignOperator,
              factory.getPointerKeyForObjectCatalog(other));
      getSystem()
          .newSideEffect(
              new DictFieldsOperator(other, receiver, pc),
              factory.getPointerKeyForObjectCatalog(other));
    }

    /** Applies {@link #copyInto} to each dictionary the update's argument may be. */
    private final class DictArgumentOperator extends UnaryOperator<PointsToSetVariable> {
      private final InstanceKey receiver;
      private final int pc;
      private final Set<InstanceKey> seen = HashSetFactory.make();

      private DictArgumentOperator(InstanceKey receiver, int pc) {
        this.receiver = receiver;
        this.pc = pc;
      }

      @Override
      public byte evaluate(PointsToSetVariable l, PointsToSetVariable r) {
        if (r.getValue() != null)
          r.getValue()
              .foreach(
                  i -> {
                    InstanceKey other = getSystem().getInstanceKey(i);
                    if (seen.add(other)) copyInto(other, receiver);
                  });
        return NOT_CHANGED;
      }

      @Override
      public int hashCode() {
        return receiver.hashCode() * 31 + pc;
      }

      @Override
      public boolean equals(Object o) {
        return o instanceof DictArgumentOperator other
            && other.receiver.equals(receiver)
            && other.pc == pc;
      }

      @Override
      public String toString() {
        return "dict update argument at " + pc + " into " + receiver;
      }
    }

    @Override
    public int hashCode() {
      return (node.hashCode() * 31 + pc) * 31 + kind.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof DictMethodOperator other
          && other.node.equals(node)
          && other.pc == pc
          && other.kind == kind;
    }

    @Override
    public String toString() {
      return "dict " + kind + " at " + pc + " in " + node;
    }
  }

  /**
   * Flows every catalogued field of one dictionary key into the field of the same name of another
   * as the catalog's names arrive (wala/ML#997). Equal for the same (from, to) pair.
   */
  private final class DictFieldsOperator extends UnaryOperator<PointsToSetVariable> {
    private final InstanceKey from;
    private final InstanceKey to;
    private final int pc;

    private DictFieldsOperator(InstanceKey from, InstanceKey to, int pc) {
      this.from = from;
      this.to = to;
      this.pc = pc;
    }

    @Override
    public byte evaluate(PointsToSetVariable l, PointsToSetVariable catalog) {
      if (catalog.getValue() == null) return NOT_CHANGED;
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IClassHierarchy cha = getClassHierarchy();
      catalog
          .getValue()
          .foreach(
              c -> {
                InstanceKey nameKey = getSystem().getInstanceKey(c);
                if (!(nameKey instanceof ConstantKey<?> constant)) return;
                Object value = constant.getValue();
                if (!(value instanceof String) && !(value instanceof Number)) return;
                IField f = resolveRootField(cha, value.toString());
                if (f == null) return;
                getSystem()
                    .newConstraint(
                        factory.getPointerKeyForInstanceField(to, f),
                        assignOperator,
                        factory.getPointerKeyForInstanceField(from, f));
              });
      return NOT_CHANGED;
    }

    @Override
    public int hashCode() {
      return from.hashCode() * 31 + to.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof DictFieldsOperator other
          && other.from.equals(from)
          && other.to.equals(to);
    }

    @Override
    public String toString() {
      return "dict update at " + pc + ": " + from + " into " + to;
    }
  }

  /**
   * Mints the result of a text read once its receiver is known (see {@code processTextRead}): a
   * string constant for a file's {@code read} or {@code readline}, and a fresh list holding a
   * string constant under index {@code 0} for a file's {@code readlines} or a string's {@code
   * splitlines} and {@code split}. The list is populated as a literal list is (a catalog entry and
   * a numbered field), so every element reader sees the string.
   */
  public final class TextReadOperator extends UnaryOperator<PointsToSetVariable> {
    public enum Kind {
      FILE_TEXT,
      FILE_LINES,
      STRING_PIECES
    }

    private static final String TEXT = "<text>";
    private final CGNode node;
    private final int pc;
    private final PointerKey resultKey;
    private final Kind kind;
    private boolean contributed;

    private TextReadOperator(CGNode node, int pc, PointerKey resultKey, Kind kind) {
      this.node = node;
      this.pc = pc;
      this.resultKey = resultKey;
      this.kind = kind;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (contributed || rhs.getValue() == null) return NOT_CHANGED;
      java.util.List<InstanceKey> receivers = new java.util.ArrayList<>();
      rhs.getValue().foreach(i -> receivers.add(getSystem().getInstanceKey(i)));
      apply(receivers.toArray(new InstanceKey[0]));
      return NOT_CHANGED;
    }

    /**
     * Applies the read once some receiver matches its kind.
     *
     * @param receivers The receiver's instance keys.
     */
    void apply(InstanceKey[] receivers) {
      if (contributed || receivers == null) return;
      boolean applies = false;
      for (InstanceKey receiver : receivers) if (receiverMatches(receiver)) applies = true;
      if (!applies) return;
      contributed = true;
      logger.fine(() -> "text read at " + pc + " in " + node + ": " + kind + " applies.");
      InstanceKey text = getInstanceKeyForConstant(PythonTypes.string, TEXT);
      if (kind == Kind.FILE_TEXT) {
        getSystem().newConstraint(resultKey, text);
        return;
      }
      InstanceKey list =
          getInstanceKeyForAllocation(node, TypedSiteReference.at(pc, PythonTypes.list));
      if (list == null) return;
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IField zero = resolveRootField(getClassHierarchy(), "0");
      if (zero == null) return;
      getSystem().newConstraint(resultKey, list);
      getSystem()
          .newConstraint(
              factory.getPointerKeyForObjectCatalog(list),
              getInstanceKeyForConstant(PythonLanguage.Python.getConstantType(0), 0));
      getSystem().newConstraint(factory.getPointerKeyForInstanceField(list, zero), text);
      logger.fine(() -> "text read at " + pc + " in " + node + ": " + kind + " -> " + list);
    }

    private boolean receiverMatches(InstanceKey receiver) {
      if (kind == Kind.STRING_PIECES)
        return receiver instanceof ConstantKey
            && ((ConstantKey<?>) receiver).getValue() instanceof String;
      IClassHierarchy cha = getClassHierarchy();
      IClass fileClass = cha.lookupClass(PythonTypes.file);
      return fileClass != null && cha.isSubclassOf(receiver.concreteType(), fileClass);
    }

    @Override
    public int hashCode() {
      return (node.hashCode() * 31 + pc) * 31 + kind.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof TextReadOperator
          && ((TextReadOperator) o).node.equals(node)
          && ((TextReadOperator) o).pc == pc
          && ((TextReadOperator) o).kind == kind;
    }

    @Override
    public String toString() {
      return "text read " + kind + " at " + pc + " in " + node;
    }
  }

  /**
   * Reads a negative constant subscript of a tuple, {@code t[-k]}, as element {@code n - k} of each
   * tuple {@code t} may be, where {@code n} is that tuple's length (wala/ML#988). A tuple's
   * elements are fields named by their index from {@code 0}, so the ordinary field read names a
   * field no tuple has, and the read's result was empty. The length is taken from the tuple's
   * allocation: the constant-index writes to the fresh tuple in the allocating method, which must
   * be exactly {@code 0} through {@code n - 1}. A tuple whose length is not known that way, and any
   * key that is not a tuple, contributes nothing, as before. A list is left alone, since its length
   * can change after it is built.
   */
  public final class NegativeSubscriptOperator extends UnaryOperator<PointsToSetVariable> {
    private final PointerKey resultKey;
    private final int index;
    private final Set<InstanceKey> read = HashSetFactory.make();

    private NegativeSubscriptOperator(PointerKey resultKey, int index) {
      this.resultKey = resultKey;
      this.index = index;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() != null) rhs.getValue().foreach(i -> read(getSystem().getInstanceKey(i)));
      return NOT_CHANGED;
    }

    /**
     * Adds the constraint reading element {@code n + index} of the given key into the result, if
     * the key is a tuple of known length {@code n} and the element exists.
     *
     * @param key A key the subscripted object may be.
     */
    private void read(InstanceKey key) {
      if (!read.add(key)) return;
      if (!(key instanceof AllocationSiteInNode asin)) return;
      IClassHierarchy cha = getClassHierarchy();
      IClass tupleClass = cha.lookupClass(PythonTypes.tuple);
      if (tupleClass == null || !key.concreteType().equals(tupleClass)) return;
      int length = knownTupleLength(asin);
      int element = length + index;
      if (length < 0 || element < 0) return;
      IField field = resolveRootField(cha, Integer.toString(element));
      if (field == null) return;
      logger.fine(() -> "negative subscript " + index + " of " + key + " reads element " + element);
      getSystem()
          .newConstraint(
              resultKey,
              assignOperator,
              ((AstPointerKeyFactory) getPointerKeyFactory())
                  .getPointerKeyForInstanceField(key, field));
    }

    @Override
    public int hashCode() {
      return resultKey.hashCode() * 31 + index;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof NegativeSubscriptOperator other
          && resultKey.equals(other.resultKey)
          && index == other.index;
    }

    @Override
    public String toString() {
      return "negative subscript " + index + " into " + resultKey;
    }
  }

  /**
   * Reads the order-free contents of each object a constant subscript {@code xs[i]} may be into the
   * read's result, except an object whose exact prefix covers {@code i}: a tuple concatenation
   * whose left operand is a tuple literal at least {@code i + 1} long, whose element {@code i} is
   * the literal's (see {@link #exactListOperationPrefixes}). The ordinary read takes that element
   * from its numbered field. Every other object's appended and operation contents are read as the
   * read of an unknown position always read them.
   */
  public final class ConstantIndexContentsReadOperator extends UnaryOperator<PointsToSetVariable> {
    private final PointerKey resultKey;
    private final int index;
    private final Set<InstanceKey> read = HashSetFactory.make();

    private ConstantIndexContentsReadOperator(PointerKey resultKey, int index) {
      this.resultKey = resultKey;
      this.index = index;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() != null) rhs.getValue().foreach(i -> read(getSystem().getInstanceKey(i)));
      return NOT_CHANGED;
    }

    /**
     * Adds the constraints reading the given key's order-free contents into the result, unless the
     * key's exact prefix covers the index.
     *
     * @param key A key the subscripted object may be.
     */
    private void read(InstanceKey key) {
      if (!read.add(key)) return;
      if (key instanceof AllocationSiteInNode asin) {
        Integer prefix =
            exactListOperationPrefixes.get(
                Pair.make(asin.getNode(), asin.getSite().getProgramCounter()));
        if (prefix != null && index < prefix) return;
      }
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IClassHierarchy cha = getClassHierarchy();
      for (String name : List.of(LIST_APPEND_CONTENTS_FIELD, LIST_OPERATION_CONTENTS_FIELD)) {
        IField field = resolveRootField(cha, name);
        if (field == null) continue;
        getSystem()
            .newConstraint(
                resultKey, assignOperator, factory.getPointerKeyForInstanceField(key, field));
      }
    }

    @Override
    public int hashCode() {
      return resultKey.hashCode() * 31 + index;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ConstantIndexContentsReadOperator other
          && resultKey.equals(other.resultKey)
          && index == other.index;
    }

    @Override
    public String toString() {
      return "constant index " + index + " contents read into " + resultKey;
    }
  }

  /**
   * Reads every element of each list or tuple a subscripted object may be into the result of a read
   * whose index is a loop variable (wala/ML#993). The elements are read through the collection's
   * catalog of field names as the names arrive, the way iteration and the list operations read
   * them, so elements written after the read is registered still reach it.
   */
  public final class UnknownIndexReadOperator extends UnaryOperator<PointsToSetVariable> {
    private final PointerKey resultKey;
    private final int pc;
    private final Set<InstanceKey> read = HashSetFactory.make();

    private UnknownIndexReadOperator(PointerKey resultKey, int pc) {
      this.resultKey = resultKey;
      this.pc = pc;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      // A side effect: the elements reach the result through the element copies registered here.
      if (rhs.getValue() != null) rhs.getValue().foreach(i -> read(getSystem().getInstanceKey(i)));
      return NOT_CHANGED;
    }

    /**
     * Registers the read of every element of the given key, if it is a list or a tuple.
     *
     * @param key A key the subscripted object may be.
     */
    private void read(InstanceKey key) {
      if (!read.add(key)) return;
      IClassHierarchy cha = getClassHierarchy();
      IClass list = cha.lookupClass(PythonTypes.list);
      IClass tuple = cha.lookupClass(PythonTypes.tuple);
      IClass type = key.concreteType();
      if (!(list != null && cha.isSubclassOf(type, list))
          && !(tuple != null && cha.isSubclassOf(type, tuple))) return;
      logger.fine(() -> "unknown-index read of every element of " + key + " into " + resultKey);
      getSystem()
          .newSideEffect(
              new ElementCopyOperator(key, resultKey, pc),
              ((AstPointerKeyFactory) getPointerKeyFactory()).getPointerKeyForObjectCatalog(key));
    }

    @Override
    public int hashCode() {
      return resultKey.hashCode() * 31 + pc;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof UnknownIndexReadOperator other
          && resultKey.equals(other.resultKey)
          && pc == other.pc;
    }

    @Override
    public String toString() {
      return "unknown-index read at " + pc + " into " + resultKey;
    }
  }

  /**
   * The length of a tuple: an {@code *args} pack's recorded length, else the length read off its
   * allocation by {@link #tupleLength(AllocationSiteInNode)}.
   *
   * @param asin The tuple's allocation.
   * @return The length, or {@code -1} when neither determines it.
   */
  private int knownTupleLength(AllocationSiteInNode asin) {
    Integer packed = packLengths.get(asin);
    return packed != null ? packed : tupleLength(asin);
  }

  /**
   * The length of a tuple, read off its allocation: the constant-index property writes to the fresh
   * tuple in the allocating method, which must be exactly {@code 0} through {@code n - 1}
   * (wala/ML#988).
   *
   * @param asin The tuple's allocation.
   * @return The length, or {@code -1} when the allocation does not determine it.
   */
  private static int tupleLength(AllocationSiteInNode asin) {
    CGNode node = asin.getNode();
    IR ir = node.getIR();
    if (ir == null || node.getDU() == null) return -1;
    // A tuple built by an operation (`(a,) + (b,)`, wala/ML#960) is allocated at a site that is not
    // a `new` of this IR, and `IR.getNew` throws rather than answering null for such a site.
    boolean allocatedHere = false;
    for (Iterator<NewSiteReference> sites = ir.iterateNewSites(); sites.hasNext(); )
      if (sites.next().equals(asin.getSite())) allocatedHere = true;
    if (!allocatedHere) return -1;
    SSANewInstruction alloc = ir.getNew(asin.getSite());
    if (alloc == null) return -1;
    return tupleElementCount(ir.getSymbolTable(), node.getDU(), alloc.getDef());
  }

  /**
   * The length of a tuple literal, a value a {@code new} of a tuple in the same IR defines, read
   * off its constant-index writes as {@link #tupleLength(AllocationSiteInNode)} reads them.
   *
   * @param symtab The symbol table of the IR that may define the value.
   * @param du The IR's def-use information.
   * @param vn The value number.
   * @return The length, or {@code -1} when the value is no tuple literal of a known length.
   */
  private static int literalTupleLength(SymbolTable symtab, DefUse du, int vn) {
    if (vn <= 0 || du == null) return -1;
    if (!(du.getDef(vn) instanceof SSANewInstruction alloc)
        || !alloc.getNewSite().getDeclaredType().equals(PythonTypes.tuple)) return -1;
    return tupleElementCount(symtab, du, vn);
  }

  /**
   * Counts a fresh tuple's elements by its constant-index property writes, which must be exactly
   * {@code 0} through {@code n - 1} (wala/ML#988).
   *
   * @param symtab The symbol table of the IR allocating the tuple.
   * @param du The IR's def-use information.
   * @param tupleVn The tuple's value number.
   * @return The length, or {@code -1} when the writes do not determine it.
   */
  private static int tupleElementCount(SymbolTable symtab, DefUse du, int tupleVn) {
    Set<Integer> indices = HashSetFactory.make();
    for (Iterator<SSAInstruction> uses = du.getUses(tupleVn); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      if (!(use instanceof AstPropertyWrite write) || write.getObjectRef() != tupleVn) continue;
      if (!symtab.isConstant(write.getMemberRef())) return -1;
      Object member = symtab.getConstantValue(write.getMemberRef());
      // Every writer of a tuple's elements (the translator's literals, the `zip` and `enumerate`
      // summaries) names them by an Integer constant.
      if (!(member instanceof Integer index) || index < 0) return -1;
      indices.add(index);
    }
    for (int i = 0; i < indices.size(); i++) if (!indices.contains(i)) return -1;
    return indices.size();
  }

  /**
   * The result of a binary operator over an array (wala/ML#1009), attached to an operand: each
   * operand key of a type {@link #freshBinaryOpResultTypes} names yields a fresh key of the mapped
   * type at the operator's instruction index, added to the result's points-to set by an instance
   * constraint. Keys of other types contribute nothing.
   */
  public final class ArrayOperationOperator extends UnaryOperator<PointsToSetVariable> {
    private final CGNode node;
    private final int pc;
    private final PointerKey resultKey;
    private final Map<TypeReference, TypeReference> types;

    private ArrayOperationOperator(
        CGNode node, int pc, PointerKey resultKey, Map<TypeReference, TypeReference> types) {
      this.node = node;
      this.pc = pc;
      this.resultKey = resultKey;
      this.types = types;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() == null) return NOT_CHANGED;
      rhs.getValue().foreach(i -> contribute(getSystem().getInstanceKey(i)));
      return NOT_CHANGED;
    }

    /**
     * Adds the fresh result an operand key yields, if its type is named, to the result's points-to
     * set.
     *
     * @param key An operand's instance key.
     */
    private void contribute(InstanceKey key) {
      TypeReference resultType = types.get(key.concreteType().getReference());
      if (resultType == null) return;
      InstanceKey fresh = getInstanceKeyForAllocation(node, TypedSiteReference.at(pc, resultType));
      if (fresh == null) return;
      getSystem().newConstraint(resultKey, fresh);
      attachArrayAttributes(node, pc, fresh);
    }

    @Override
    public int hashCode() {
      return (node.hashCode() * 31 + pc) * 31 + resultKey.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ArrayOperationOperator other
          && node.equals(other.node)
          && pc == other.pc
          && resultKey.equals(other.resultKey);
    }

    @Override
    public String toString() {
      return "array operation@" + pc;
    }
  }

  /**
   * The result of a list repetition or concatenation (wala/ML#960), attached to ONE operand: for
   * each list or tuple key flowing into that operand, when the operation's rule holds against the
   * other operand's contents, a fresh key of the same type allocated at the binop's instruction
   * index joins the result, and the operand key's elements, every field its object catalog names
   * plus its own appended and operation contents, flow into the fresh key's synthetic {@value
   * #LIST_OPERATION_CONTENTS_FIELD} field, which non-constant subscripts, iteration and {@code zip}
   * read and no shape reader interprets. Keys of other types contribute nothing, and a key already
   * contributed is not registered again.
   *
   * <p>The rule: {@code ADD} fires only when the other operand also carries a list or tuple (list
   * plus list); a list beside a tensor or an ndarray is that object's own addition, whose result is
   * no list. {@code MUL} fires only when the other operand carries neither a list or tuple nor a
   * tensor-like value (a tensor or an ndarray, by their model packages): repetition takes an
   * integer, and a list times a tensor is the tensor's multiplication.
   *
   * <p>A key declined because the other operand has not met the rule yet is kept pending and
   * retried when that operand's set changes. The converse is the monotone limit: a repetition
   * contributed while the other operand held only integers is not retracted if that operand later
   * grows a tensor member.
   */
  public final class ListOperationOperator extends UnaryOperator<PointsToSetVariable> {
    private final CGNode node;
    private final int pc;
    private final PointerKey resultKey;
    private final IBinaryOpInstruction.IOperator operator;
    private final int operandIndex;

    /** The other operand's key, or {@code null} for a constant (the multiplier). */
    private final PointerKey otherKey;

    /** The other operand's contents when invariant or implicit, else {@code null}. */
    private final InstanceKey[] otherInvariant;

    private final Set<InstanceKey> contributed = HashSetFactory.make();

    /** Keys of this operand declined so far because the other operand had not met the rule. */
    private final Set<InstanceKey> pending = HashSetFactory.make();

    private ListOperationOperator(
        CGNode node,
        int pc,
        PointerKey resultKey,
        IBinaryOpInstruction.IOperator operator,
        int operandIndex,
        PointerKey otherKey,
        InstanceKey[] otherInvariant) {
      this.node = node;
      this.pc = pc;
      this.resultKey = resultKey;
      this.operator = operator;
      this.operandIndex = operandIndex;
      this.otherKey = otherKey;
      this.otherInvariant = otherInvariant;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() == null) return NOT_CHANGED;
      if (otherKey != null && otherKey.equals(rhs.getPointerKey())) {
        // The other operand grew: retry this operand's keys declined before.
        for (InstanceKey key : new java.util.ArrayList<>(pending)) contribute(key);
        return NOT_CHANGED;
      }
      rhs.getValue().foreach(i -> contribute(getSystem().getInstanceKey(i)));
      return NOT_CHANGED;
    }

    /**
     * Adds the fresh key an operand key contributes to the result, if any and if the operation's
     * rule holds against the other operand, to the result's points-to set by an instance
     * constraint, which is not an edge of the flow graph. A key seen before contributes nothing
     * again.
     *
     * @param key An operand's instance key.
     */
    private void contribute(InstanceKey key) {
      if (!isListOrTuple(key) || contributed.contains(key)) return;
      if (!ruleHolds()) {
        pending.add(key); // retried when the other operand's set changes
        return;
      }
      pending.remove(key);
      contributed.add(key);
      InstanceKey fresh =
          getInstanceKeyForAllocation(
              node, TypedSiteReference.at(pc, key.concreteType().getReference()));
      if (fresh == null) return;
      getSystem().newConstraint(resultKey, fresh);
      copyElements(key, fresh);
      // The left literal's elements keep their positions in the result as well.
      Integer prefix = exactListOperationPrefixes.get(Pair.make(node, pc));
      if (prefix != null && operandIndex == 0) copyPrefix(key, fresh, prefix);
    }

    /**
     * Flows element {@code i} of {@code from} into element {@code i} of {@code to} for each {@code
     * i} below {@code length}.
     */
    private void copyPrefix(InstanceKey from, InstanceKey to, int length) {
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IClassHierarchy cha = getClassHierarchy();
      for (int i = 0; i < length; i++) {
        IField field = resolveRootField(cha, Integer.toString(i));
        if (field == null) continue;
        getSystem()
            .newConstraint(
                factory.getPointerKeyForInstanceField(to, field),
                assignOperator,
                factory.getPointerKeyForInstanceField(from, field));
      }
    }

    /** Whether the operation's rule holds against the other operand's current contents. */
    private boolean ruleHolds() {
      boolean otherList = false;
      boolean otherTensorLike = false;
      Iterable<InstanceKey> contents =
          otherInvariant == null ? null : java.util.Arrays.asList(otherInvariant);
      if (contents == null && otherKey != null) {
        PointsToSetVariable v = getSystem().findOrCreatePointsToSet(otherKey);
        java.util.List<InstanceKey> keys = new java.util.ArrayList<>();
        if (v.getValue() != null)
          v.getValue().foreach(i -> keys.add(getSystem().getInstanceKey(i)));
        contents = keys;
      }
      if (contents != null)
        for (InstanceKey ik : contents) {
          if (isListOrTuple(ik)) otherList = true;
          else if (isTensorLike(ik)) otherTensorLike = true;
        }
      if (operator == IBinaryOpInstruction.Operator.ADD) return otherList;
      return !otherList && !otherTensorLike;
    }

    private boolean isListOrTuple(InstanceKey key) {
      IClassHierarchy cha = getClassHierarchy();
      IClass type = key.concreteType();
      return cha.isSubclassOf(type, cha.lookupClass(PythonTypes.list))
          || cha.isSubclassOf(type, cha.lookupClass(PythonTypes.tuple));
    }

    /** A tensor or an ndarray, by the model packages that declare them. */
    private boolean isTensorLike(InstanceKey key) {
      String name = key.concreteType().getName().toString();
      return name.startsWith("Ltensorflow/") || name.startsWith("Lnumpy/");
    }

    /**
     * Flows every catalogued field of {@code from}, and its appended and operation contents, into
     * {@code to}'s operation-contents field. The catalog is read by a side effect with value
     * equality on (from, to), so re-evaluation registers nothing twice.
     */
    private void copyElements(InstanceKey from, InstanceKey to) {
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IClassHierarchy cha = getClassHierarchy();
      IField contents = resolveRootField(cha, LIST_OPERATION_CONTENTS_FIELD);
      IField appended = resolveRootField(cha, LIST_APPEND_CONTENTS_FIELD);
      if (contents == null || appended == null) return;
      PointerKey contentsKey = factory.getPointerKeyForInstanceField(to, contents);
      logger.fine(() -> "list operation at " + pc + " in " + node + ": " + to + " from " + from);
      // The operand's own appended contents and operation contents flow through unchanged, so a
      // chain of operations, or an operation over an appended list, keeps every element.
      getSystem()
          .newConstraint(
              contentsKey, assignOperator, factory.getPointerKeyForInstanceField(from, contents));
      getSystem()
          .newConstraint(
              contentsKey, assignOperator, factory.getPointerKeyForInstanceField(from, appended));
      getSystem()
          .newSideEffect(
              new ElementCopyOperator(from, contentsKey, pc),
              factory.getPointerKeyForObjectCatalog(from));
    }

    @Override
    public int hashCode() {
      return ((node.hashCode() * 31 + pc) * 31 + resultKey.hashCode()) * 31 + operandIndex;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ListOperationOperator
          && ((ListOperationOperator) o).node.equals(node)
          && ((ListOperationOperator) o).pc == pc
          && ((ListOperationOperator) o).resultKey.equals(resultKey)
          && ((ListOperationOperator) o).operandIndex == operandIndex;
    }

    @Override
    public String toString() {
      return "list operation result at " + pc + " in " + node + " (operand " + operandIndex + ")";
    }
  }

  /**
   * Binds a call's arguments to a target's {@code *args} and {@code **kwargs} formals, and unpacks
   * a call's starred arguments into the target's formals (wala/ML#991). Before, a {@code *args}
   * formal bound the one positional argument at its own index, every later one bound the formals
   * after it (the {@code **kwargs} formal first) or nothing, a keyword naming no formal was
   * dropped, and a starred argument bound its whole iterable to one formal; so a wrapper {@code def
   * w(*args, **kwargs): return f(*args, **kwargs)} forwarded only its first argument.
   *
   * <p>The positional arguments from the {@code *args} formal's index on are packed into a tuple,
   * and the keywords naming no formal into a dict, each allocated at the call (one per caller
   * context) and bound to its formal. The site is the call's alone, so two targets of one
   * polymorphic call with different {@code *args} indices share one pack and their elements merge
   * (sound, and rare). A starred argument's elements bind the target's positional formals from its
   * slot on, by the iterable's element index, those past a {@code *args} formal joining its pack;
   * elements of unknown index (appended or operation contents) may bind any of them. A {@code **}
   * argument binds each named formal its dict has a key for, and its whole dict to a {@code
   * **kwargs} formal. The arguments after a starred one keep the alignment by slot they had
   * (wala/ML#751), and an iterable whose elements are not indexed fields (a list built by an
   * operation, wala/ML#960) binds through its unknown-index contents only.
   */
  private final class StarArguments {
    private final CGNode caller;
    private final PythonInvokeInstruction call;
    private final CGNode target;
    private final InstanceKey[][] constParams;
    private final int varargs;
    private final int keywords;
    private final int firstStarred;
    private final boolean forwardingBody;
    private final int formals;
    private InstanceKey pack;
    private InstanceKey keywordPack;

    private StarArguments(
        CGNode caller, PythonInvokeInstruction call, CGNode target, InstanceKey[][] constParams) {
      this.caller = caller;
      this.call = call;
      this.target = target;
      this.constParams = constParams;
      IMethod method = target.getMethod();
      this.varargs =
          method instanceof StarFormalDeclaration declaration
              ? declaration.getVarargsParameter()
              : -1;
      this.keywords =
          method instanceof StarFormalDeclaration declaration
              ? declaration.getKeywordsParameter()
              : -1;
      // A synthesized forwarding body (a method or callable trampoline) receives a starred or
      // `**` argument intact and forwards it still marked, so the target it forwards to unpacks
      // it; unpacking it here as well would spread it twice. A synthesized constructor forwards
      // its own formals to `__init__` positionally and by `__init__`'s names, so a starred or `**`
      // argument is unpacked at the call to it instead.
      this.forwardingBody =
          method instanceof PythonSummarizedFunction
              && !(method instanceof PythonConstructorFunction);
      this.firstStarred = this.forwardingBody ? -1 : call.firstStarredPosition();
      this.formals = method.getNumberOfParameters();
      // A `*args` formal is a tuple even when no argument reaches it.
      if (this.varargs >= 0 && this.varargs < this.formals) this.pack();
    }

    /**
     * Binds a positional slot that the starred and packed arguments govern.
     *
     * @param slot The positional slot.
     * @return {@code true} iff the slot was bound here.
     */
    private boolean bindPositional(int slot) {
      if (slot == this.firstStarred && slot > 0) {
        this.unpack(slot);
        return true;
      }
      if (this.varargs >= 0 && slot >= this.varargs && slot > 0) {
        InstanceKey tuple = this.pack();
        if (tuple == null) return false;
        if (this.constParams != null
            && slot < this.constParams.length
            && this.constParams[slot] != null)
          for (InstanceKey element : this.constParams[slot])
            this.packElement(tuple, slot - this.varargs, element);
        else
          this.packElement(
              tuple,
              slot - this.varargs,
              getPointerKeyForLocal(this.caller, this.call.getUse(slot)));
        return true;
      }
      return false;
    }

    /**
     * Whether a method is a library summary body as the bypass selector presents one, a method read
     * from a summary file, as opposed to a program method or a trampoline or constructor the front
     * end synthesizes, which forward the program's arguments as the program passed them. A summary
     * body wrapped another way (a synthesized constructor copying a summary's statements) is not
     * recognized, and its pack keeps the call's padded length.
     *
     * @param method The method.
     * @return {@code true} iff the method is a library summary body.
     */
    private boolean isLibrarySummaryBody(IMethod method) {
      return method instanceof SummarizedMethodWithNames
          && !(method instanceof PythonSummarizedFunction);
    }

    /** The positional pack, allocated and bound to the {@code *args} formal on first use. */
    private InstanceKey pack() {
      if (this.pack == null) {
        this.pack =
            getInstanceKeyForAllocation(
                this.caller, TypedSiteReference.at(this.call.iIndex(), PythonTypes.tuple));
        if (this.pack != null) {
          // Without a starred argument, the pack holds exactly the positional arguments from the
          // `*args` formal's index on. A library summary's call is the exception: it pads its
          // arguments to a fixed count (`strategy.run`, `tf.while_loop`), so its positions are not
          // the program's.
          int length =
              this.call.firstStarredPosition() < 0 && !isLibrarySummaryBody(this.caller.getMethod())
                  ? Math.max(
                      0, this.call.getNumberOfPositionalParameters() - Math.max(1, this.varargs))
                  : -1;
          packLengths.merge(this.pack, length, (a, b) -> a.equals(b) ? a : -1);
          getSystem()
              .newConstraint(getPointerKeyForLocal(this.target, this.varargs + 1), this.pack);
        }
      }
      return this.pack;
    }

    private void packElement(InstanceKey tuple, int index, Object value) {
      IField field = resolveRootField(getClassHierarchy(), Integer.toString(index));
      if (field == null) return;
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      getSystem()
          .newConstraint(
              factory.getPointerKeyForObjectCatalog(tuple),
              getInstanceKeyForConstant(PythonLanguage.Python.getConstantType(index), index));
      PointerKey slot = factory.getPointerKeyForInstanceField(tuple, field);
      if (value instanceof InstanceKey element) getSystem().newConstraint(slot, element);
      else getSystem().newConstraint(slot, assignOperator, (PointerKey) value);
    }

    /** Unpacks the starred argument at a slot into the target's formals and pack. */
    private void unpack(int slot) {
      UnaryOperator<PointsToSetVariable> operator = new UnpackOperator(slot);
      if (this.constParams != null
          && slot < this.constParams.length
          && this.constParams[slot] != null)
        for (InstanceKey iterable : this.constParams[slot])
          ((UnpackOperator) operator).read(iterable);
      else
        getSystem()
            .newSideEffect(operator, getPointerKeyForLocal(this.caller, this.call.getUse(slot)));
    }

    /** The target's formal at a positional index, or {@code null} past the plain formals. */
    private PointerKey positionalFormal(int index) {
      int limit = this.varargs >= 0 ? this.varargs : this.formals;
      return index > 0 && index < limit ? getPointerKeyForLocal(this.target, index + 1) : null;
    }

    /**
     * Unpacks a starred argument's iterables: each element at index {@code m} binds the formal at
     * {@code slot + m}, or the pack past a {@code *args} formal; an element of unknown index binds
     * every formal from the slot on and the pack.
     */
    private final class UnpackOperator extends UnaryOperator<PointsToSetVariable> {
      private final int slot;
      private final Set<InstanceKey> read = HashSetFactory.make();

      private UnpackOperator(int slot) {
        this.slot = slot;
      }

      @Override
      public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
        // A side effect: the elements reach the formals through the constraints registered here.
        if (rhs.getValue() != null)
          rhs.getValue().foreach(i -> read(getSystem().getInstanceKey(i)));
        return NOT_CHANGED;
      }

      private void read(InstanceKey iterable) {
        if (!this.read.add(iterable)) return;
        IClassHierarchy cha = getClassHierarchy();
        AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
        for (String unindexed :
            new String[] {LIST_APPEND_CONTENTS_FIELD, LIST_OPERATION_CONTENTS_FIELD}) {
          IField field = resolveRootField(cha, unindexed);
          if (field == null) continue;
          PointerKey contents = factory.getPointerKeyForInstanceField(iterable, field);
          int limit = varargs >= 0 ? varargs : formals;
          for (int index = this.slot; index < limit; index++) {
            PointerKey formal = positionalFormal(index);
            if (formal != null) getSystem().newConstraint(formal, assignOperator, contents);
          }
          if (varargs >= 0) {
            InstanceKey tuple = pack();
            IField packContents = resolveRootField(cha, LIST_OPERATION_CONTENTS_FIELD);
            if (tuple != null && packContents != null) {
              getSystem()
                  .newConstraint(
                      factory.getPointerKeyForObjectCatalog(tuple),
                      getInstanceKeyForConstant(PythonTypes.string, LIST_OPERATION_CONTENTS_FIELD));
              getSystem()
                  .newConstraint(
                      factory.getPointerKeyForInstanceField(tuple, packContents),
                      assignOperator,
                      contents);
            }
          }
        }
        getSystem()
            .newSideEffect(
                new UnaryOperator<PointsToSetVariable>() {
                  private final Set<Integer> bound = HashSetFactory.make();

                  @Override
                  public byte evaluate(PointsToSetVariable l, PointsToSetVariable catalog) {
                    if (catalog.getValue() == null) return NOT_CHANGED;
                    catalog
                        .getValue()
                        .foreach(
                            c -> {
                              InstanceKey name = getSystem().getInstanceKey(c);
                              if (!(name instanceof ConstantKey<?> constant)
                                  || !(constant.getValue() instanceof Number number)) return;
                              int m = number.intValue();
                              if (m < 0 || !bound.add(m)) return;
                              IField source = resolveRootField(cha, Integer.toString(m));
                              if (source == null) return;
                              PointerKey element =
                                  factory.getPointerKeyForInstanceField(iterable, source);
                              int index = slot + m;
                              if (varargs >= 0 && index >= varargs) {
                                InstanceKey tuple = pack();
                                if (tuple != null) packElement(tuple, index - varargs, element);
                              } else {
                                PointerKey formal = positionalFormal(index);
                                if (formal != null)
                                  getSystem().newConstraint(formal, assignOperator, element);
                              }
                            });
                    return NOT_CHANGED;
                  }

                  @Override
                  public int hashCode() {
                    return System.identityHashCode(this);
                  }

                  @Override
                  public boolean equals(Object o) {
                    return this == o;
                  }

                  @Override
                  public String toString() {
                    return "unpack elements of " + iterable + " at slot " + slot;
                  }
                },
                factory.getPointerKeyForObjectCatalog(iterable));
      }

      @Override
      public int hashCode() {
        return (caller.hashCode() * 31 + call.iIndex()) * 31 + target.hashCode() + this.slot;
      }

      @Override
      public boolean equals(Object o) {
        return o instanceof UnpackOperator other
            && Objects.equals(other.outer(), StarArguments.this.identity())
            && other.slot == this.slot;
      }

      private Object outer() {
        return StarArguments.this.identity();
      }

      @Override
      public String toString() {
        return "unpack starred slot " + this.slot + " at " + call.iIndex() + " in " + caller;
      }
    }

    /** The (caller, call, target) triple identifying these bindings. */
    private Object identity() {
      return java.util.List.of(this.caller, this.call.iIndex(), this.target);
    }

    /**
     * Binds a {@code **} argument (the keyword the front end names {@code null}): each named formal
     * the dict has a key for, and the whole dict to a {@code **kwargs} formal.
     *
     * @return {@code true} iff the keyword was a {@code **} argument and is bound here.
     */
    private boolean bindDoubleStarred(String argName, int src, InstanceKey[] constants) {
      if (!"null".equals(argName) || this.forwardingBody) return false;
      PointerKey dict = getPointerKeyForLocal(this.caller, src);
      // A dict built in the caller's own body is an invariant argument: its keys arrive as
      // constants and its pointer key is implicit, which no constraint may name (wala/ML#925).
      boolean implicit = constants == null && getSystem().isImplicit(dict);
      if (this.keywords >= 0) {
        PointerKey formal = getPointerKeyForLocal(this.target, this.keywords + 1);
        if (constants != null)
          for (InstanceKey key : constants) getSystem().newConstraint(formal, key);
        else if (!implicit) getSystem().newConstraint(formal, assignOperator, dict);
      }
      Map<String, PointerKey> named = HashMapFactory.make();
      for (int index = 1; index < this.formals; index++) {
        if (index == this.varargs || index == this.keywords) continue;
        String[] names = this.target.getIR().getLocalNames(0, index + 1);
        if (names == null) continue;
        for (String name : names)
          if (name != null) named.put(name, getPointerKeyForLocal(this.target, index + 1));
      }
      if (named.isEmpty()) return true;
      DoubleStarOperator operator = new DoubleStarOperator(named);
      if (constants != null) for (InstanceKey key : constants) operator.read(key);
      else if (!implicit) getSystem().newSideEffect(operator, dict);
      return true;
    }

    /** Binds each named formal from the field of that name of each dict a {@code **} value is. */
    private final class DoubleStarOperator extends UnaryOperator<PointsToSetVariable> {
      private final Map<String, PointerKey> named;
      private final Set<InstanceKey> read = HashSetFactory.make();

      private DoubleStarOperator(Map<String, PointerKey> named) {
        this.named = named;
      }

      @Override
      public byte evaluate(PointsToSetVariable l, PointsToSetVariable r) {
        // A side effect: the entries reach the formals through the constraints registered here.
        if (r.getValue() != null) r.getValue().foreach(i -> read(getSystem().getInstanceKey(i)));
        return NOT_CHANGED;
      }

      private void read(InstanceKey dict) {
        if (!this.read.add(dict)) return;
        IClassHierarchy cha = getClassHierarchy();
        AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
        for (Map.Entry<String, PointerKey> entry : this.named.entrySet()) {
          IField field = resolveRootField(cha, entry.getKey());
          if (field != null)
            getSystem()
                .newConstraint(
                    entry.getValue(),
                    assignOperator,
                    factory.getPointerKeyForInstanceField(dict, field));
        }
      }

      @Override
      public int hashCode() {
        return Objects.hash(identity(), this.named.keySet());
      }

      @Override
      public boolean equals(Object o) {
        return o instanceof DoubleStarOperator other
            && Objects.equals(other.outer(), identity())
            && other.named.keySet().equals(this.named.keySet());
      }

      private Object outer() {
        return identity();
      }

      @Override
      public String toString() {
        return "unpack ** at " + call.iIndex() + " in " + caller;
      }
    }

    /** Collects a keyword that names no formal into the {@code **kwargs} formal's dict. */
    private void packKeyword(String argName, InstanceKey[] constants, int src) {
      if (this.keywords < 0 || "null".equals(argName)) return;
      if (this.keywordPack == null) {
        this.keywordPack =
            getInstanceKeyForAllocation(
                this.caller, TypedSiteReference.at(this.call.iIndex(), PythonTypes.dict));
        if (this.keywordPack == null) return;
        getSystem()
            .newConstraint(getPointerKeyForLocal(this.target, this.keywords + 1), this.keywordPack);
      }
      IField field = resolveRootField(getClassHierarchy(), argName);
      if (field == null) return;
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      getSystem()
          .newConstraint(
              factory.getPointerKeyForObjectCatalog(this.keywordPack),
              getInstanceKeyForConstant(PythonLanguage.Python.getConstantType(argName), argName));
      PointerKey slot = factory.getPointerKeyForInstanceField(this.keywordPack, field);
      if (constants != null)
        for (InstanceKey constant : constants) getSystem().newConstraint(slot, constant);
      else getSystem().newConstraint(slot, assignOperator, getPointerKeyForLocal(this.caller, src));
    }
  }

  private static IField resolveRootField(IClassHierarchy cha, String name) {
    return cha.resolveField(
        FieldReference.findOrCreate(
            PythonTypes.Root, Atom.findOrCreateUnicodeAtom(name), PythonTypes.Root));
  }

  /**
   * Flows the elements of each iterable a starred literal element may be into the literal's
   * operation contents (wala/ML#989): a list's or tuple's catalogued fields and its own appended
   * and operation contents. Any other iterable (an ndarray, a tensor shape, a generator) is stored
   * itself, as the literal stored it before, since its elements are not fields the analysis knows.
   */
  public final class StarredElementOperator extends UnaryOperator<PointsToSetVariable> {
    private final PointerKey target;
    private final PointerKey value;
    private final int pc;
    private final Set<InstanceKey> read = HashSetFactory.make();
    private boolean keptWhole;

    private StarredElementOperator(PointerKey target, PointerKey value, int pc) {
      this.target = target;
      this.value = value;
      this.pc = pc;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      // A side effect: the elements reach the literal through the constraints registered here.
      if (rhs.getValue() != null) rhs.getValue().foreach(i -> read(getSystem().getInstanceKey(i)));
      return NOT_CHANGED;
    }

    /**
     * Registers the flow of one iterable's elements into the literal.
     *
     * @param iterable A key the starred value may be.
     */
    private void read(InstanceKey iterable) {
      if (!read.add(iterable)) return;
      IClassHierarchy cha = getClassHierarchy();
      IClass list = cha.lookupClass(PythonTypes.list);
      IClass tuple = cha.lookupClass(PythonTypes.tuple);
      IClass type = iterable.concreteType();
      if (!(list != null && cha.isSubclassOf(type, list))
          && !(tuple != null && cha.isSubclassOf(type, tuple))) {
        // Kept whole through an assignment from the starred value, so the tensor dataflow, which
        // walks assignments, sees it as it did when the literal stored the value as one element.
        // The assignment carries the value's whole points-to set: when it mixes a collection with
        // a non-collection, the collection is stored whole as well as unpacked (a superset).
        if (!keptWhole) {
          keptWhole = true;
          getSystem().newConstraint(target, assignOperator, value);
        }
        return;
      }
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IField contents = resolveRootField(cha, LIST_OPERATION_CONTENTS_FIELD);
      IField appended = resolveRootField(cha, LIST_APPEND_CONTENTS_FIELD);
      if (contents != null)
        getSystem()
            .newConstraint(
                target, assignOperator, factory.getPointerKeyForInstanceField(iterable, contents));
      if (appended != null)
        getSystem()
            .newConstraint(
                target, assignOperator, factory.getPointerKeyForInstanceField(iterable, appended));
      getSystem()
          .newSideEffect(
              new ElementCopyOperator(iterable, target, pc),
              factory.getPointerKeyForObjectCatalog(iterable));
    }

    @Override
    public int hashCode() {
      return target.hashCode() * 31 + pc;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof StarredElementOperator other
          && target.equals(other.target)
          && pc == other.pc;
    }

    @Override
    public String toString() {
      return "starred element at " + pc + " into " + target;
    }
  }

  /**
   * Attaches to a fresh array the methods its type's model attaches per instance (wala/ML#1009):
   * each named attribute receives a fresh instance of the summary class the model's own allocators
   * attach, allocated at the same site, so a slice or an arithmetic result dispatches its methods
   * as an array the summaries allocate does. Per-instance attachment mirrors the NumPy summaries;
   * wala/ML#551 replaces both with class-level methods.
   *
   * @param node The node allocating the array.
   * @param pc The allocation's site.
   * @param array The fresh array.
   */
  private void attachArrayAttributes(CGNode node, int pc, InstanceKey array) {
    Map<String, TypeReference> attributes =
        freshArrayAttributes.get(array.concreteType().getReference());
    if (attributes == null) return;
    AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
    IClassHierarchy cha = getClassHierarchy();
    attributes.forEach(
        (name, type) -> {
          IField f = resolveRootField(cha, name);
          InstanceKey method = getInstanceKeyForAllocation(node, TypedSiteReference.at(pc, type));
          if (f != null && method != null)
            getSystem().newConstraint(factory.getPointerKeyForInstanceField(array, f), method);
        });
  }

  /**
   * Flows every catalogued field of a list key into a contents field as the catalog's names arrive
   * (wala/ML#960). Equal for the same (from, contents) pair, so the propagation system keeps one.
   */
  private final class ElementCopyOperator extends UnaryOperator<PointsToSetVariable> {
    private final InstanceKey from;
    private final PointerKey contentsKey;
    private final int pc;

    private ElementCopyOperator(InstanceKey from, PointerKey contentsKey, int pc) {
      this.from = from;
      this.contentsKey = contentsKey;
      this.pc = pc;
    }

    @Override
    public byte evaluate(PointsToSetVariable l, PointsToSetVariable catalog) {
      if (catalog.getValue() == null) return NOT_CHANGED;
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IClassHierarchy cha = getClassHierarchy();
      catalog
          .getValue()
          .foreach(
              c -> {
                InstanceKey nameKey = getSystem().getInstanceKey(c);
                if (!(nameKey instanceof ConstantKey)) return;
                Object value = ((ConstantKey<?>) nameKey).getValue();
                // A literal's element fields are named by integer constants, a dictionary's by
                // strings; both name a field.
                if (!(value instanceof String) && !(value instanceof Number)) return;
                String name = value.toString();
                IField f = resolveRootField(cha, name);
                if (f == null) return;
                logger.fine(
                    () ->
                        "list operation at "
                            + pc
                            + ": field "
                            + name
                            + " of "
                            + from
                            + " into "
                            + contentsKey);
                getSystem()
                    .newConstraint(
                        contentsKey,
                        assignOperator,
                        factory.getPointerKeyForInstanceField(from, f));
              });
      return NOT_CHANGED;
    }

    @Override
    public int hashCode() {
      return from.hashCode() * 31 + contentsKey.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ElementCopyOperator
          && ((ElementCopyOperator) o).from.equals(from)
          && ((ElementCopyOperator) o).contentsKey.equals(contentsKey);
    }

    @Override
    public String toString() {
      return "list-operation elements of " + from + " into " + contentsKey;
    }
  }

  /**
   * The assignment from a {@code slice} call's receiver to its result (wala/ML#916): every key
   * passes through except one of a {@link #freshSliceResultTypes fresh type}, which becomes the
   * fresh allocation of the type it maps to at the call. Equal for the same call, so the constraint
   * is idempotent like an assignment.
   *
   * <p>Every key of a fresh type in the receiver's set maps to the ONE allocation of that type at
   * this call, so two distinct tensors reaching the slice through a merge are one object after it.
   * That is a deliberate precision choice, not an accident: the result's shape and dtype are
   * resolved from the call itself (the slice operation over the whole receiver), never from the
   * individual key, so per-key allocations would multiply objects without sharpening a reading.
   *
   * <p>When the instance-key factory declines the allocation (a {@code null} key), the receiver's
   * key passes through instead: the result then aliases its receiver at that key exactly as before
   * this change, a decline rather than a failure inside the solver.
   */
  private final class SliceResultOperator extends UnaryOperator<PointsToSetVariable> {
    private final CGNode caller;
    private final int pc;

    private SliceResultOperator(CGNode caller, int pc) {
      this.caller = caller;
      this.pc = pc;
    }

    /** The literal list and tuple keys already sliced, mapped to their slice's key. */
    private final Map<InstanceKey, InstanceKey> sliced = HashMapFactory.make();

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() == null) return NOT_CHANGED;
      Map<TypeReference, TypeReference> fresh = freshSliceResultTypes;
      MutableIntSet out = IntSetUtil.make();
      rhs.getValue()
          .foreach(
              i -> {
                InstanceKey key = getSystem().getInstanceKey(i);
                InstanceKey slice = sliceOfLiteral(key);
                if (slice != null) {
                  out.add(getSystem().findOrCreateIndexForInstanceKey(slice));
                  return;
                }
                TypeReference type = fresh.get(key.concreteType().getReference());
                InstanceKey allocation =
                    type != null
                        ? getInstanceKeyForAllocation(caller, TypedSiteReference.at(pc, type))
                        : null;
                // The slice is an array of the receiver's kind, with its methods (wala/ML#1009).
                if (allocation != null) attachArrayAttributes(caller, pc, allocation);
                // A declined allocation (null) falls back to the receiver's own key: the type came
                // off an existing key, so the class resolves and this is not expected to happen,
                // but a null inside the solver would take the whole analysis down (the wala/ML#925
                // class) where aliasing the receiver merely loses this change at one key.
                out.add(
                    allocation == null
                        ? i
                        : getSystem().findOrCreateIndexForInstanceKey(allocation));
              });
      return lhs.addAll(out) ? CHANGED : NOT_CHANGED;
    }

    /**
     * The slice of a list or tuple literal by constant bounds, as a fresh collection of the same
     * type allocated at the slice's call holding only the elements in range (wala/ML#993), where
     * the result aliased the whole receiver and so carried the elements the slice drops. The fresh
     * collection is populated as a literal is, a catalog entry and a numbered field per element.
     * The receiver's appended and operation contents, whose positions are not known, flow into the
     * slice's as well. A list can grow past its literal after it is built, so once any such
     * contents reach the receiver, the elements past the upper bound are kept as well: a bound
     * counted from the end no longer names the literal's elements.
     *
     * @param key A key the sliced object may be.
     * @return The slice's key, or {@code null} when the key is not a list or tuple literal of known
     *     length, the bounds are not constants, or the step is not one; the caller then aliases the
     *     receiver as before.
     */
    private InstanceKey sliceOfLiteral(InstanceKey key) {
      if (sliced.containsKey(key)) return sliced.get(key);
      InstanceKey slice = null;
      int[] range = literalRange(key);
      if (range != null) {
        slice =
            getInstanceKeyForAllocation(
                caller, TypedSiteReference.at(pc, key.concreteType().getReference()));
        if (slice != null) populate(key, slice, range[0], range[1], range[2]);
      }
      sliced.put(key, slice);
      return slice;
    }

    /**
     * The {@code [start, stop)} range of literal elements this slice keeps of a key, with the
     * literal's length, or {@code null} when the key or the bounds do not determine it.
     */
    private int[] literalRange(InstanceKey key) {
      if (!(key instanceof AllocationSiteInNode asin)) return null;
      IClassHierarchy cha = getClassHierarchy();
      IClass list = cha.lookupClass(PythonTypes.list);
      IClass tuple = cha.lookupClass(PythonTypes.tuple);
      IClass type = key.concreteType();
      if (!(list != null && cha.isSubclassOf(type, list))
          && !(tuple != null && cha.isSubclassOf(type, tuple))) return null;
      // A list a summary allocates stands in for a value whose length the summary does not model
      // (`tf.unstack` writes one stand-in piece at a few constant indices), so its constant writes
      // are not its length, and slicing it as a literal mints a shorter list than the program's
      // (wala/ML#993). A summary's tuple keeps its length: `zip` and `enumerate` pairs are fixed.
      if (list != null
          && cha.isSubclassOf(type, list)
          && asin.getNode().getMethod().isWalaSynthetic()) return null;
      int length = knownTupleLength(asin);
      if (length < 0) return null;
      SSAInstruction inst = caller.getIR().getInstructions()[pc];
      if (!(inst instanceof PythonInvokeInstruction call)
          || call.getNumberOfPositionalParameters() < 3) return null;
      SymbolTable symtab = caller.getIR().getSymbolTable();
      Integer lower = bound(symtab, call, 2, 0);
      Integer upper = bound(symtab, call, 3, length);
      Integer step = bound(symtab, call, 4, 1);
      if (lower == null || upper == null || step == null || step != 1) return null;
      int start = clamp(lower < 0 ? lower + length : lower, length);
      int stop = clamp(upper < 0 ? upper + length : upper, length);
      return new int[] {start, Math.max(start, stop), length};
    }

    /** A slice bound: its integer constant, the default when it is absent or {@code None}. */
    private Integer bound(SymbolTable symtab, PythonInvokeInstruction call, int use, int dflt) {
      if (use >= call.getNumberOfPositionalParameters()) return dflt;
      int vn = call.getUse(use);
      if (vn <= 0 || symtab.isNullConstant(vn)) return dflt;
      if (!symtab.isConstant(vn)) return null;
      Object value = symtab.getConstantValue(vn);
      if (value == null) return dflt;
      return value instanceof Number number ? number.intValue() : null;
    }

    private static int clamp(int index, int length) {
      return Math.max(0, Math.min(index, length));
    }

    private void populate(InstanceKey from, InstanceKey slice, int start, int stop, int length) {
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IClassHierarchy cha = getClassHierarchy();
      logger.fine(() -> "slice [" + start + ", " + stop + ") of " + from + " at " + pc);
      copyElements(from, slice, start, stop, start, factory, cha);
      IField contents = resolveRootField(cha, LIST_OPERATION_CONTENTS_FIELD);
      IField appended = resolveRootField(cha, LIST_APPEND_CONTENTS_FIELD);
      if (contents == null || appended == null) return;
      PointerKey sliceContents = factory.getPointerKeyForInstanceField(slice, contents);
      PointerKey fromContents = factory.getPointerKeyForInstanceField(from, contents);
      PointerKey fromAppended = factory.getPointerKeyForInstanceField(from, appended);
      getSystem().newConstraint(sliceContents, assignOperator, fromContents);
      getSystem().newConstraint(sliceContents, assignOperator, fromAppended);
      if (stop < length) {
        // Once the receiver may have grown, keep the literal's elements past the upper bound too.
        UnaryOperator<PointsToSetVariable> grown =
            new UnaryOperator<>() {
              private boolean widened;

              @Override
              public byte evaluate(PointsToSetVariable l, PointsToSetVariable r) {
                if (widened || r.getValue() == null || r.getValue().isEmpty()) return NOT_CHANGED;
                widened = true;
                copyElements(from, slice, stop, length, start, factory, cha);
                return NOT_CHANGED;
              }

              @Override
              public int hashCode() {
                return from.hashCode() * 31 + slice.hashCode();
              }

              @Override
              public boolean equals(Object o) {
                return this == o;
              }

              @Override
              public String toString() {
                return "slice widening of " + from + " into " + slice;
              }
            };
        getSystem().newSideEffect(grown, fromContents);
        getSystem().newSideEffect(grown, fromAppended);
      }
    }

    /**
     * Copies elements {@code [lo, hi)} of a literal into the slice, renumbered from the slice's
     * start: element {@code j} becomes the slice's element {@code j - start}.
     */
    private void copyElements(
        InstanceKey from,
        InstanceKey slice,
        int lo,
        int hi,
        int start,
        AstPointerKeyFactory factory,
        IClassHierarchy cha) {
      for (int j = lo; j < hi; j++) {
        int index = j - start;
        IField source = resolveRootField(cha, Integer.toString(j));
        IField target = resolveRootField(cha, Integer.toString(index));
        if (source == null || target == null) continue;
        getSystem()
            .newConstraint(
                factory.getPointerKeyForObjectCatalog(slice),
                getInstanceKeyForConstant(PythonLanguage.Python.getConstantType(index), index));
        getSystem()
            .newConstraint(
                factory.getPointerKeyForInstanceField(slice, target),
                assignOperator,
                factory.getPointerKeyForInstanceField(from, source));
      }
    }

    @Override
    public int hashCode() {
      return caller.hashCode() * 31 + pc;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof SliceResultOperator
          && ((SliceResultOperator) o).caller.equals(caller)
          && ((SliceResultOperator) o).pc == pc;
    }

    @Override
    public String toString() {
      return "slice result at " + pc + " in " + caller;
    }
  }

  /**
   * Returns the {@link PointerKey} to bind for the given target's dispatched (position-0)
   * parameter, honoring a receiver filter the target's context carries (<a
   * href="https://github.com/wala/ML/issues/679">wala/ML#679</a>). The filter applies only to
   * trampoline targets: their first parameter <em>is</em> the dispatched object the context is
   * keyed on. A real method body inheriting a per-receiver context has the same context item, but
   * its first parameter is the function object, which the receiver filter would wrongly empty.
   *
   * @param target The callee {@link CGNode}.
   * @param dflt The unfiltered {@link PointerKey} for the target's first parameter.
   * @return A {@link FilteredPointerKey} restricted to the context's receiver when the target is a
   *     trampoline whose context supplies a parameter-0 filter; otherwise {@code dflt}.
   */
  private PointerKey getReceiverFilteredPointerKey(CGNode target, PointerKey dflt) {
    if (!(target.getMethod().getDeclaringClass() instanceof PythonInstanceMethodTrampoline))
      return dflt;
    TypeFilter filter = (TypeFilter) target.getContext().get(ContextKey.PARAMETERS[0]);
    if (filter != null && !filter.isRootFilter())
      return getFilteredPointerKeyForLocal(target, 1, filter);
    return dflt;
  }

  /**
   * Returns true iff the given {@link MethodReference} is a "callable" method, i.e., a method that
   * is used to implement the __call__ functionality of a callable object.
   *
   * @param methodReference The {@link MethodReference} in question.
   * @return True iff the given {@link MethodReference} is a "callable" method.
   */
  private static boolean isCallable(MethodReference methodReference) {
    String name = methodReference.getDeclaringClass().getName().toString();
    return name.endsWith(CALLABLE_METHOD_NAME)
        || name.endsWith(CALLABLE_METHOD_NAME_FOR_KERAS_MODELS);
  }

  /**
   * Returns true iff the given {@link PointerKey} points to at least one instance whose concrete
   * type equals {@link PythonTypes#object}.
   *
   * @param pointerKey The {@link PointerKey} in question.
   * @return True iff the given {@link PointerKey} points to at least one object whose concrete type
   *     equals {@link PythonTypes#object}.,
   */
  protected boolean refersToAnObject(PointerKey pointerKey) {
    PointerAnalysis<InstanceKey> pointerAnalysis = this.getPointerAnalysis();
    OrdinalSet<InstanceKey> pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);

    for (InstanceKey instanceKey : pointsToSet) {
      IClass concreteType = instanceKey.concreteType();
      TypeReference reference = concreteType.getReference();

      // If it's an "object" method.
      if (reference.equals(PythonTypes.object)) return true;

      // Handle synthetic classes (e.g., from XML summaries) which inherit from object
      // but are not functions or trampolines.
      IClassHierarchy cha = pointerAnalysis.getClassHierarchy();
      IClass objClass = cha.lookupClass(PythonTypes.object);
      IClass trampClass = cha.lookupClass(PythonTypes.trampoline);

      if (objClass != null && cha.isSubclassOf(concreteType, objClass)) {
        if (trampClass == null || !cha.isSubclassOf(concreteType, trampClass)) {
          // Do not treat generated trampoline classes (which contain '$') as generic objects
          if (!concreteType.getName().toString().contains("$")) {
            return true;
          }
        }
      }
    }

    return false;
  }

  @Override
  public PythonConstraintVisitor makeVisitor(CGNode node) {
    return new PythonConstraintVisitor(this, node);
  }

  public static class PythonInterestingVisitor extends AstInterestingVisitor
      implements PythonInstructionVisitor {
    public PythonInterestingVisitor(int vn) {
      super(vn);
    }

    @Override
    public void visitBinaryOp(final SSABinaryOpInstruction instruction) {
      bingo = true;
    }

    @Override
    public void visitPythonBinaryOp(PythonBinaryOpInstruction binop) {
      bingo = true;
    }

    @Override
    public void visitPythonInvoke(PythonInvokeInstruction inst) {
      bingo = true;
    }
  }

  @Override
  protected InterestingVisitor makeInterestingVisitor(CGNode node, int vn) {
    return new PythonInterestingVisitor(vn);
  }

  /**
   * The name of the global holding a defaulted parameter's materialized default: {@code
   * <entity>_defaults_<i>} for a Python default, {@code <entity>_click_defaults_<i>} for a {@code
   * @click.option} default (wala/ML#971). A synthesized constructor's trailing formal i mirrors
   * {@code __init__}'s parameter i + 1, and the default globals are written under {@code
   * __init__}'s entity name, so the lookup follows that mapping (wala/ML#762). Every other target
   * reads its own entity's global.
   *
   * @param target The callee.
   * @param i The zero-based parameter index in the callee's formals.
   * @param infix {@code "_defaults_"} or {@code "_click_defaults_"}.
   * @return The global's name, without the {@code global } prefix.
   */
  private static String defaultsGlobalName(CGNode target, int i, String infix) {
    return target.getMethod() instanceof PythonConstructorFunction
        ? target.getMethod().getDeclaringClass().getName()
            + "/"
            + INIT_METHOD_NAME
            + infix
            + (i + 1)
        : target.getMethod().getDeclaringClass().getName() + infix + i;
  }

  /**
   * Resolves a global by name to its field on {@code Root}.
   *
   * @param name The global's name, without the {@code global } prefix.
   * @return The field; never {@code null}, since globals are dynamic fields created on lookup.
   */
  private IField resolveGlobal(String name) {
    FieldReference global =
        FieldReference.findOrCreate(
            PythonTypes.Root, Atom.findOrCreateUnicodeAtom("global " + name), PythonTypes.Root);
    return getClassHierarchy().resolveField(global);
  }

  /**
   * Whether a key is a materialized {@code @click.option} default (wala/ML#971): a constant key of
   * the {@link PythonTypes#clickDefault} or {@link PythonTypes#clickDefaultString} class. Such a
   * key carries its value like any constant key, so a shape or dtype reader takes it as it takes a
   * Python default, but a comparison fold must decline it: the default is the value of the one
   * invocation that passes no option, and every other invocation the command line admits binds the
   * parameter otherwise, so a guard folded from it prunes arms the program runs.
   *
   * @param key The instance key.
   * @return {@code true} for a click default's constant key.
   */
  public static boolean isClickDefault(InstanceKey key) {
    if (!(key instanceof ConstantKey)) return false;
    TypeReference type = key.concreteType().getReference();
    return type.equals(PythonTypes.clickDefault) || type.equals(PythonTypes.clickDefaultString);
  }

  /**
   * Binds a {@code @click.option} default to its parameter under a constant key of the click
   * default's own class (wala/ML#971): each constant key in the click-defaults global becomes the
   * same value under {@link PythonTypes#clickDefaultString} for a string and {@link
   * PythonTypes#clickDefault} otherwise, so {@link #isClickDefault(InstanceKey)} tells it from an
   * ordinary constant while its value reads unchanged. A member that is not a constant key (a list
   * or tuple default) passes through as it is.
   *
   * <p>The operator is wired on every unpassed defaulted parameter, not only on those with a click
   * option, because the reader cannot tell them apart by name: a parameter without a click option
   * has an empty click-defaults global, so the operator contributes nothing there.
   */
  private final class ClickDefaultOperator extends UnaryOperator<PointsToSetVariable> {
    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() == null) return NOT_CHANGED;
      MutableIntSet out = IntSetUtil.make();
      rhs.getValue()
          .foreach(
              i -> {
                InstanceKey key = getSystem().getInstanceKey(i);
                if (!(key instanceof ConstantKey) || isClickDefault(key)) {
                  out.add(i);
                  return;
                }
                Object value = ((ConstantKey<?>) key).getValue();
                TypeReference type =
                    value instanceof String
                        ? PythonTypes.clickDefaultString
                        : PythonTypes.clickDefault;
                out.add(
                    getSystem()
                        .findOrCreateIndexForInstanceKey(getInstanceKeyForConstant(type, value)));
              });
      return lhs.addAll(out) ? CHANGED : NOT_CHANGED;
    }

    @Override
    public int hashCode() {
      return ClickDefaultOperator.class.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ClickDefaultOperator;
    }

    @Override
    public String toString() {
      return "click default";
    }
  }

  /**
   * Whether an instance key is the None constant. {@code None} is one global {@link ConstantKey}
   * whose concrete type is {@code Root}, so the core builder's null-receiver filter, which asks the
   * language whether the key's <em>type</em> is the null type, does not recognize it.
   *
   * @param key The instance key.
   * @return {@code true} iff the key is the None constant.
   */
  public static boolean isNoneConstant(InstanceKey key) {
    return key instanceof ConstantKey && ((ConstantKey<?>) key).getValue() == null;
  }

  /**
   * The pointer analysis makes no field key for the None constant (wala/ML#964): an attribute write
   * on {@code None} raises at run time and so does a read, so neither carries a value. The core put
   * and get operators call this method and skip a {@code null} key. Without this, a write whose
   * receiver set merely includes {@code None} (a function object local also bound to {@code None},
   * a defaulted parameter) lands its value on the one global {@code None} key, and every wildcard
   * element read over a container that may be {@code None} (a {@code zip} or {@code for} over a
   * list that is {@code None} on one arm) reads that value back as an element of an unrelated
   * container. The heap model, which ModRef and other consumers read, still resolves the key
   * through the factory, so no {@code null} leaves the analysis.
   *
   * @param I The instance key.
   * @param field The field.
   * @return The pointer key, or {@code null} for the None constant.
   */
  @Override
  public PointerKey getPointerKeyForInstanceField(InstanceKey I, IField field) {
    if (isNoneConstant(I)) return null;
    return super.getPointerKeyForInstanceField(I, field);
  }

  /**
   * A mapping of script names to wildcard imports included in the script.
   *
   * @return A mapping of script names to wildcard imports included in the corresponding script.
   */
  protected Map<String, Deque<MethodReference>> getScriptToWildcardImports() {
    return scriptToWildcardImports;
  }
}
