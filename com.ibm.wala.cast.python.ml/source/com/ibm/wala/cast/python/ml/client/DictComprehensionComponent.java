package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.withoutClientOnlyKeys;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAArrayStoreInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPhiInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Pair;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.logging.Logger;

/**
 * A constant-key read of a dict built by a comprehension over another dict's {@code items()} (<a
 * href="https://github.com/wala/ML/issues/1071">wala/ML#1071</a>).
 *
 * <p>The front end lowers {@code {k: f(v) for k, v in d.items()}} inline, in the enclosing
 * function: it allocates the dict, iterates {@code d.items()}, reads {@code k} and {@code v} off
 * each item, and writes {@code f(v)} under {@code k}. The key is not a constant, so the write names
 * no field a constant-key read can find, and {@code v} is the join of every value of {@code d}, so
 * the value written is derived from all of them. A read of key {@code "a"} therefore re-derives the
 * written value with {@code v} bound to {@code d["a"]} alone: an explicit {@link
 * TensorGenerator.ValueBinding}, carried in every memo key the derivation issues.
 *
 * <p>The re-derivation applies only where it is sound, so a comprehension falls back to the read's
 * ordinary result when the written value depends on the key, when it reaches the value through a
 * merge, when the value escapes into the heap on the way, or when {@code d} has no known component
 * at the key. These are decided from the comprehension's code. One more condition is decided once
 * the resolution converges: every generator in the derivation that consumes the value must have
 * read it through the binding, rather than through its points-to set, which the binding does not
 * change. A derivation that fails it declines to the unknown-marked element. An {@code if} filter
 * does not prevent the re-derivation, since a read that succeeds at run time implies its key was
 * kept.
 */
public class DictComprehensionComponent extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(DictComprehensionComponent.class.getName());

  /**
   * A recognized comprehension: the write {@code D[k] = e} in {@code node}, where {@code k} and
   * {@code v} are the components of an item of {@code source.items()}.
   *
   * @param node The node holding the comprehension.
   * @param key The key variable {@code k}.
   * @param value The value variable {@code v}.
   * @param stored The value written, {@code e}.
   * @param source The dict iterated, {@code d}.
   * @param valueDependents The values in {@code node} computed from {@code v}, including it.
   */
  record Comprehension(
      CGNode node, int key, int value, int stored, int source, Set<Integer> valueDependents) {}

  /** Per builder, the comprehension each dict allocation is built by, if any. */
  private static final Map<PropagationCallGraphBuilder, Map<InstanceKey, Optional<Comprehension>>>
      COMPREHENSIONS = Collections.synchronizedMap(new WeakHashMap<>());

  /** The key read. */
  private final String key;

  /** The comprehensions the read's dict may be built by, each with its binding for the key. */
  private final List<Pair<Comprehension, ValueBinding>> bindings;

  /**
   * Constructs the generator of a key read.
   *
   * @param source The read's {@link PointsToSetVariable}.
   * @param key The key read.
   * @param bindings The comprehensions the read's dict may be built by, with their bindings.
   */
  private DictComprehensionComponent(
      PointsToSetVariable source, String key, List<Pair<Comprehension, ValueBinding>> bindings) {
    super(source);
    this.key = key;
    this.bindings = bindings;
  }

  /**
   * Constructs the generator a bound re-derivation runs in, anchored on the comprehension's node.
   *
   * @param node The comprehension's node.
   * @param binding The binding.
   */
  private DictComprehensionComponent(CGNode node, ValueBinding binding) {
    super(node);
    this.key = null;
    this.bindings = List.of();
    this.bindValue(binding);
  }

  /**
   * Clears the per-builder memo of recognized comprehensions.
   *
   * @param builder The builder whose memo to clear.
   */
  static void clearComprehensions(PropagationCallGraphBuilder builder) {
    COMPREHENSIONS.remove(builder);
  }

  /**
   * A recognized constant-key read of a comprehension-built dict: the key and, for each
   * comprehension the dict may be built by, its binding for the key.
   *
   * @param key The key read.
   * @param bindings The comprehensions with their bindings.
   */
  record Read(String key, List<Pair<Comprehension, ValueBinding>> bindings) {}

  /**
   * Constructs the generator of a recognized key read.
   *
   * @param source The read's {@link PointsToSetVariable}.
   * @param read The recognized read.
   */
  DictComprehensionComponent(PointsToSetVariable source, Read read) {
    this(source, read.key(), read.bindings());
  }

  /**
   * Recognizes a constant-key read of a comprehension-built dict, or returns {@code null} when the
   * read is not one, or falls back. Outside an analysis run there is no resolver to validate the
   * re-derivation at convergence (wala/ML#753), so the read falls back there too.
   *
   * @param node The node holding the read.
   * @param read The read.
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return The recognized read, or {@code null}.
   */
  static Read recognizeRead(
      CGNode node, PythonPropertyRead read, PropagationCallGraphBuilder builder) {
    if (WorklistTypeResolver.active(builder) == null) return null;
    String key = constantStringMember(node, read);
    if (key == null) return null;
    List<Pair<Comprehension, ValueBinding>> bindings = new ArrayList<>();
    for (InstanceKey dict : objects(builder, node, read.getObjectRef())) {
      Comprehension comprehension = comprehensionOf(builder, dict);
      // A dict that may be built otherwise holds what the read's ordinary resolution finds.
      if (comprehension == null) return null;
      ValueBinding binding =
          new ValueBinding(
              comprehension.node(),
              comprehension.value(),
              comprehension.node(),
              comprehension.source(),
              List.of(key));
      // A source with no known component at the key leaves nothing to bind to.
      if (!componentResolves(builder, binding)) return null;
      bindings.add(Pair.make(comprehension, binding));
    }
    if (bindings.isEmpty()) return null;
    LOGGER.fine(() -> "Comprehension read of \"" + key + "\" binds " + bindings + ".");
    return new Read(key, bindings);
  }

  /**
   * The objects a value may be: its points-to set, or, when that is empty, the allocation defining
   * it in its node. A dict allocated in the reading node often has an implicit pointer key whose
   * points-to set is not materialized.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The node defining the value.
   * @param valueNumber The value.
   * @return The objects.
   */
  private static Iterable<InstanceKey> objects(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber) {
    PointerKey key =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, valueNumber);
    var pointsTo = withoutClientOnlyKeys(builder.getPointerAnalysis().getPointsToSet(key));
    if (pointsTo != null && !pointsTo.isEmpty()) return pointsTo;
    if (node.getDU().getDef(valueNumber) instanceof SSANewInstruction allocation) {
      InstanceKey allocated =
          builder
              .getPointerAnalysis()
              .getHeapModel()
              .getInstanceKeyForAllocation(node, allocation.getNewSite());
      if (allocated != null) return List.of(allocated);
    }
    return List.of();
  }

  /**
   * The constant string a read's member is, if any.
   *
   * @param node The node holding the read.
   * @param read The read.
   * @return The string, or {@code null} when the member is not a string constant.
   */
  private static String constantStringMember(CGNode node, PythonPropertyRead read) {
    IR ir = node.getIR();
    if (ir == null) return null;
    SymbolTable st = ir.getSymbolTable();
    int member = read.getMemberRef();
    return st.isStringConstant(member) ? st.getStringValue(member) : null;
  }

  /**
   * Whether the binding's source has a known component at its path.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param binding The binding.
   * @return {@code true} iff the source stores or models a component there.
   */
  private static boolean componentResolves(
      PropagationCallGraphBuilder builder, ValueBinding binding) {
    var component = boundComponentPointsToSet(builder, binding);
    if (component != null && !component.isEmpty()) return true;
    return boundComponentProvider(builder, binding) != null;
  }

  /**
   * The comprehension a dict allocation is built by, memoized per builder.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param dict The allocation.
   * @return The comprehension, or {@code null} when the allocation is not a dict built by a
   *     recognized comprehension whose written value can be re-derived per key.
   */
  static Comprehension comprehensionOf(PropagationCallGraphBuilder builder, InstanceKey dict) {
    Map<InstanceKey, Optional<Comprehension>> memo =
        COMPREHENSIONS.computeIfAbsent(builder, b -> Collections.synchronizedMap(new HashMap<>()));
    Optional<Comprehension> known = memo.get(dict);
    if (known == null) {
      known = Optional.ofNullable(recognize(builder, dict));
      memo.put(dict, known);
    }
    return known.orElse(null);
  }

  /**
   * Recognizes the comprehension a dict allocation is built by.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param dict The allocation.
   * @return The comprehension, or {@code null}.
   */
  private static Comprehension recognize(PropagationCallGraphBuilder builder, InstanceKey dict) {
    AllocationSiteInNode allocation = getAllocationSiteInNode(dict);
    if (allocation == null) return null;
    if (!allocation.concreteType().getReference().equals(PythonTypes.dict)) return null;
    CGNode node = allocation.getNode();
    IR ir = node.getIR();
    DefUse du = node.getDU();
    if (ir == null || du == null) return null;
    SymbolTable st = ir.getSymbolTable();

    // The dict's value number, defined by its allocation.
    int dictValue = -1;
    for (SSAInstruction inst : ir.getInstructions())
      if (inst instanceof SSANewInstruction allocate
          && allocate.getNewSite().equals(allocation.getSite())) dictValue = allocate.getDef();
    if (dictValue < 0) return null;

    // Exactly one write to the dict in its node, under a key that is not a constant.
    PythonPropertyWrite write = null;
    for (Iterator<SSAInstruction> uses = du.getUses(dictValue); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      if (use instanceof PythonPropertyWrite candidate && candidate.getObjectRef() == dictValue) {
        if (write != null || st.isConstant(candidate.getMemberRef())) return null;
        write = candidate;
      }
    }
    if (write == null) return null;
    int keyValue = write.getMemberRef();
    int stored = write.getValue();

    // `k = item[0]`; `v = item[1]`.
    if (!(du.getDef(keyValue) instanceof PythonPropertyRead keyRead)
        || !isConstantIndex(st, keyRead.getMemberRef(), 0)) return null;
    int item = keyRead.getObjectRef();
    int value = -1;
    for (Iterator<SSAInstruction> uses = du.getUses(item); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      if (use instanceof PythonPropertyRead valueRead
          && valueRead.getObjectRef() == item
          && isConstantIndex(st, valueRead.getMemberRef(), 1)) {
        if (value >= 0) return null;
        value = valueRead.getDef();
      }
    }
    if (value < 0) return null;

    // `item = next(iterator)`; `iterator = iter(items)`; `items = source.items()`.
    if (!(du.getDef(item) instanceof SSAAbstractInvokeInstruction next)
        || !calls(builder, node, next, PythonTypes.NEXT_BUILTIN)
        || next.getNumberOfUses() < 2) return null;
    if (!(du.getDef(next.getUse(1)) instanceof SSAAbstractInvokeInstruction iter)
        || !calls(builder, node, iter, PythonTypes.ITER_BUILTIN)
        || iter.getNumberOfUses() < 2) return null;
    if (!(du.getDef(iter.getUse(1)) instanceof SSAAbstractInvokeInstruction items)) return null;
    if (!(du.getDef(items.getUse(0)) instanceof PythonPropertyRead itemsMethod)
        || !st.isStringConstant(itemsMethod.getMemberRef())
        || !"items".equals(st.getStringValue(itemsMethod.getMemberRef()))) return null;
    int source = itemsMethod.getObjectRef();

    Set<Integer> valueDependents = dependents(du, value);
    // A written value computed from the key differs per key in a way the binding does not model.
    if (dependents(du, keyValue).contains(stored)) return null;
    // A merge between the value and the written value selects among arms per iteration, a choice
    // the binding does not model (a conditional expression lowers to one).
    if (reachesThroughMerge(du, stored, valueDependents)) return null;
    // A value stored into the heap on its way to the written value is read back through a
    // container, whose consumer the convergence check cannot attribute to the value.
    for (int dependent : valueDependents)
      for (Iterator<SSAInstruction> uses = du.getUses(dependent); uses.hasNext(); ) {
        SSAInstruction use = uses.next();
        if (use == write) continue;
        if (use instanceof PythonPropertyWrite
            || use instanceof SSAPutInstruction
            || use instanceof SSAArrayStoreInstruction) return null;
      }

    return new Comprehension(node, keyValue, value, stored, source, valueDependents);
  }

  /**
   * Whether a value number is a constant equal to the given index.
   *
   * @param st The symbol table.
   * @param valueNumber The value number.
   * @param index The index.
   * @return {@code true} iff the value is that integer constant.
   */
  private static boolean isConstantIndex(SymbolTable st, int valueNumber, int index) {
    if (!st.isConstant(valueNumber)) return false;
    Object constant = st.getConstantValue(valueNumber);
    return constant instanceof Number number && number.intValue() == index;
  }

  /**
   * Whether an invoke may call the given builtin.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @param node The node holding the invoke.
   * @param invoke The invoke.
   * @param builtin The builtin's type.
   * @return {@code true} iff one of the invoke's targets is the builtin.
   */
  private static boolean calls(
      PropagationCallGraphBuilder builder,
      CGNode node,
      SSAAbstractInvokeInstruction invoke,
      TypeReference builtin) {
    for (CGNode target : builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite()))
      if (target.getMethod().getReference().getDeclaringClass().equals(builtin)) return true;
    return false;
  }

  /**
   * The values in a node computed from a value, the value included.
   *
   * @param du The node's def-use information.
   * @param value The value.
   * @return The value and every value defined by an instruction using one of them.
   */
  private static Set<Integer> dependents(DefUse du, int value) {
    Set<Integer> ret = new HashSet<>();
    Deque<Integer> worklist = new ArrayDeque<>();
    ret.add(value);
    worklist.add(value);
    while (!worklist.isEmpty()) {
      int current = worklist.poll();
      for (Iterator<SSAInstruction> uses = du.getUses(current); uses.hasNext(); ) {
        SSAInstruction use = uses.next();
        for (int i = 0; i < use.getNumberOfDefs(); i++)
          if (ret.add(use.getDef(i))) worklist.add(use.getDef(i));
      }
    }
    return ret;
  }

  /**
   * Whether the written value's derivation from the value passes through a φ.
   *
   * @param du The node's def-use information.
   * @param stored The written value.
   * @param valueDependents The values computed from the value.
   * @return {@code true} iff a φ computed from the value lies on the written value's derivation.
   */
  private static boolean reachesThroughMerge(DefUse du, int stored, Set<Integer> valueDependents) {
    Set<Integer> seen = new HashSet<>();
    Deque<Integer> worklist = new ArrayDeque<>();
    worklist.add(stored);
    while (!worklist.isEmpty()) {
      int current = worklist.poll();
      if (!valueDependents.contains(current) || !seen.add(current)) continue;
      SSAInstruction def = du.getDef(current);
      if (def == null) continue;
      if (def instanceof SSAPhiInstruction) return true;
      for (int i = 0; i < def.getNumberOfUses(); i++) worklist.add(def.getUse(i));
    }
    return false;
  }

  @Override
  protected Object operationDiscriminator() {
    return this.key;
  }

  @Override
  protected ShapeResult getShapeResult(PropagationCallGraphBuilder builder) {
    WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
    if (engine == null) return ShapeResult.unknown();
    ShapeResult ret = ShapeResult.bottom();
    for (Pair<Comprehension, ValueBinding> pair : this.bindings) {
      Comprehension comprehension = pair.fst;
      ValueBinding binding = pair.snd;
      this.registerCheck(engine, comprehension, binding, "shape-eval");
      if (engine.isBindingDeclined(binding)) return ShapeResult.unknown();
      DictComprehensionComponent bound =
          new DictComprehensionComponent(comprehension.node(), binding);
      ret =
          ret.union(
              bound.getShapeResult(builder, comprehension.node(), comprehension.stored(), false));
    }
    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.getShapeResult(builder).toLegacy();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
    if (engine == null) return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (Pair<Comprehension, ValueBinding> pair : this.bindings) {
      Comprehension comprehension = pair.fst;
      ValueBinding binding = pair.snd;
      this.registerCheck(engine, comprehension, binding, "dtype-eval");
      if (engine.isBindingDeclined(binding)) return EnumSet.of(DType.UNKNOWN);
      DictComprehensionComponent bound =
          new DictComprehensionComponent(comprehension.node(), binding);
      Set<DType> dtypes = bound.getDTypes(builder, comprehension.node(), comprehension.stored());
      if (dtypes == null) return EnumSet.of(DType.UNKNOWN);
      ret.addAll(dtypes);
    }
    return ret;
  }

  /**
   * Registers the convergence check of a binding this read's evaluation relies on.
   *
   * @param engine The active resolver.
   * @param comprehension The comprehension.
   * @param binding The binding.
   * @param tag The evaluation's memo tag, naming the root query to re-evaluate on a decline.
   */
  private void registerCheck(
      WorklistTypeResolver engine, Comprehension comprehension, ValueBinding binding, String tag) {
    engine.registerBindingCheck(
        binding,
        resolver -> appliesBinding(resolver, comprehension, binding),
        Pair.make(tag, evaluationKey(this)));
  }

  /**
   * The convergence check: within the bound derivation, every generator anchored on an instruction
   * consuming a value computed from the bound value read that value through the binding, no query
   * of the derivation read such a value unbound, and the bound value itself was read when the
   * written value is computed from it.
   *
   * @param resolver The converged resolver.
   * @param comprehension The comprehension.
   * @param binding The binding.
   * @return {@code true} iff the derivation applied the binding throughout.
   */
  private static boolean appliesBinding(
      WorklistTypeResolver resolver, Comprehension comprehension, ValueBinding binding) {
    CGNode node = comprehension.node();
    Set<Integer> dependents = comprehension.valueDependents();
    Set<Object> visited = new HashSet<>();
    Deque<Object> worklist = new ArrayDeque<>();
    for (Object key : allKeys(resolver, binding, node, comprehension.stored())) worklist.add(key);
    boolean consulted = false;
    while (!worklist.isEmpty()) {
      Object key = worklist.poll();
      if (!visited.add(key)) continue;
      Pair<CGNode, Integer> query = valueQueryOf(key);
      if (query != null
          && query.fst.equals(node)
          && query.snd == comprehension.value()
          && key instanceof BoundKey) consulted = true;
      Set<Integer> consumed = consumedDependents(key, node, dependents);
      boolean readBound = consumed.isEmpty();
      for (Object read : resolver.readsOf(key)) {
        Pair<CGNode, Integer> readQuery = valueQueryOf(read);
        if (readQuery != null && readQuery.fst.equals(node) && dependents.contains(readQuery.snd)) {
          // A value computed from the bound one, read without the binding.
          if (!(read instanceof BoundKey bound) || !bound.binding().equals(binding)) return false;
          if (consumed.contains(readQuery.snd)) readBound = true;
        }
        if (read instanceof BoundKey bound && bound.binding().equals(binding)) worklist.add(read);
      }
      if (!readBound) return false;
    }
    return consulted || !dependents.contains(comprehension.stored());
  }

  /**
   * The bound queries of the written value that the resolver holds.
   *
   * @param resolver The resolver.
   * @param binding The binding.
   * @param node The comprehension's node.
   * @param stored The written value.
   * @return The written value's bound shape and dtype queries that were evaluated.
   */
  private static List<Object> allKeys(
      WorklistTypeResolver resolver, ValueBinding binding, CGNode node, int stored) {
    List<Object> ret = new ArrayList<>();
    for (Object key : resolver.boundQueries(binding)) {
      Pair<CGNode, Integer> query = valueQueryOf(key);
      if (query != null && query.fst.equals(node) && query.snd == stored) ret.add(key);
    }
    return ret;
  }

  /**
   * The values computed from the bound one that a generator evaluation's anchor consumes.
   *
   * @param key A memo key.
   * @param node The comprehension's node.
   * @param dependents The values computed from the bound one.
   * @return The consumed values; empty when the key is no generator evaluation anchored in the
   *     comprehension's node.
   */
  private static Set<Integer> consumedDependents(Object key, CGNode node, Set<Integer> dependents) {
    if (!(key instanceof BoundKey bound)) return Set.of();
    if (!(bound.key() instanceof Pair<?, ?> tagged)
        || !("shape-eval".equals(tagged.fst) || "dtype-eval".equals(tagged.fst))
        || !(tagged.snd instanceof Pair<?, ?> evaluation)
        || !(evaluation.snd instanceof Pair<?, ?> anchored)
        || !(anchored.fst instanceof LocalPointerKey anchor)
        || !anchor.getNode().equals(node)) return Set.of();
    SSAInstruction def = node.getDU().getDef(anchor.getValueNumber());
    if (def == null) return Set.of();
    Set<Integer> ret = new HashSet<>();
    for (int i = 0; i < def.getNumberOfUses(); i++)
      if (dependents.contains(def.getUse(i))) ret.add(def.getUse(i));
    return ret;
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
