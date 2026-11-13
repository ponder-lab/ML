package com.ibm.wala.cast.python.ipa.callgraph;

import static com.ibm.wala.cast.python.types.PythonTypes.dict;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.set;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstGlobalWrite;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.intset.IntSet;
import com.ibm.wala.util.intset.IntSetUtil;
import com.ibm.wala.util.intset.MutableIntSet;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * What the program may do to containers that the analysis does not record among their elements: the
 * lists whose positioned elements may not be their run-time ones, and the lists, dicts and sets
 * that may hold an element under a key the analysis does not record. Read off the whole call graph
 * and the program functions it does not reach; see {@link #of}. A client's container model reads it
 * to decide whether a list's recorded positions are its run-time ones.
 *
 * @param lists The instance keys, by their index in the pointer analysis's instance key mapping, of
 *     the lists whose positioned elements it may so change.
 * @param allLists Whether it may so change any list's at all.
 * @param unappended The instance keys, by the same index, of the lists whose positioned elements it
 *     may so change other than by a call of {@code append} on the list, which the analysis records
 *     in the list's appended contents as it solves.
 * @param allUnappended Whether it may so change any list's at all other than by such a call.
 * @param open The instance keys, by the same index, of the lists, dicts and sets it may add an
 *     element to under a key the analysis does not record.
 * @param allOpen Whether it may so add an element to any list, dict or set at all.
 */
public record ContainerMutations(
    IntSet lists,
    boolean allLists,
    IntSet unappended,
    boolean allUnappended,
    IntSet open,
    boolean allOpen) {

  private static final Logger LOGGER = Logger.getLogger(ContainerMutations.class.getName());

  /**
   * The list methods that may change a list's length or the order of its elements, whose effect the
   * analysis does not record among the list's positioned elements. {@code append} is among them:
   * the analysis records an appended element under its own channel, but not that the length grew.
   */
  private static final Set<String> REPOSITIONING_LIST_METHODS =
      Set.of(
          "append",
          "extend",
          "insert",
          "pop",
          "remove",
          "clear",
          "sort",
          "reverse",
          "__setitem__",
          "__delitem__",
          "__iadd__",
          "__imul__");

  /** The type name of the builtin a {@code del} statement is lowered to a call of. */
  private static final String DELETE_BUILTIN = "Lwala/builtin/__delete__";

  /** The type name of the builtin a slice, {@code l[i:j]}, is lowered to a call of. */
  private static final String SLICE_BUILTIN = "Lwala/builtin/slice";

  /** The name a {@code del} statement's lowering reads the deletion builtin by. */
  private static final String DELETE_NAME = "__delete__";

  /**
   * The type-name prefixes of the libraries whose functions do not resize a Python list passed to
   * them, so an unresolved call of one marks nothing; see {@link #of}.
   */
  private static final List<String> NON_RESIZING_LIBRARIES = List.of("Ltensorflow", "Lnumpy");

  /**
   * The library functions named like a method that may add an element ({@link #ADDING_METHODS})
   * that resize no container, by the library module's type name and the attributes read off it:
   * {@code tf.keras.backend.update} assigns a variable (wala/ML#1182).
   */
  private static final Set<String> NON_RESIZING_LIBRARY_FUNCTIONS =
      Set.of("Ltensorflow/keras/backend/update");

  /** The builtins whose call builds a fresh container (wala/ML#1182). */
  private static final Set<String> CONTAINER_BUILTINS = Set.of("list", "dict", "set");

  /**
   * The methods that may add an element to a list, dict or set without the analysis recording it
   * among the container's element keys, a set's {@code add} among them. A list's {@code append} is
   * not: the analysis records an appended element under its own channel.
   */
  private static final Set<String> ADDING_METHODS =
      Set.of(
          "extend",
          "insert",
          "__setitem__",
          "__iadd__",
          "__imul__",
          "add",
          "update",
          "setdefault",
          "__ior__");

  /** The mutations {@link #of} is collecting. */
  private static final class Marks {

    /** See {@link #lists}. */
    final MutableIntSet lists = IntSetUtil.make();

    /** See {@link #open}. */
    final MutableIntSet open = IntSetUtil.make();

    /** See {@link #allLists}. */
    boolean allLists;

    /** See {@link #unappended}. */
    final MutableIntSet unappended = IntSetUtil.make();

    /** See {@link #allUnappended}. */
    boolean allUnappended;

    /** See {@link #allOpen}. */
    boolean allOpen;

    /**
     * Marks the containers a local may point to.
     *
     * @param pa The pointer analysis.
     * @param node The local's node.
     * @param vn The local.
     * @param adds Whether the mutation may also add an element the analysis does not record.
     */
    void mark(PointerAnalysis<InstanceKey> pa, CGNode node, int vn, boolean adds) {
      for (InstanceKey ik : pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, vn))) {
        TypeReference type = ik.concreteType().getReference();
        int index = pa.getInstanceKeyMapping().getMappedIndex(ik);
        if (type.equals(list)) {
          this.lists.add(index);
          this.unappended.add(index);
        }
        if (adds && isMutable(type)) this.open.add(index);
      }
    }

    /**
     * Marks the lists a local may point to for a call of {@code append} on it, which the analysis
     * records in the list's appended contents as it solves, so they are not {@link #unappended}.
     *
     * @param pa The pointer analysis.
     * @param node The local's node.
     * @param vn The local.
     */
    void markAppended(PointerAnalysis<InstanceKey> pa, CGNode node, int vn) {
      for (InstanceKey ik : pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, vn)))
        if (ik.concreteType().getReference().equals(list))
          this.lists.add(pa.getInstanceKeyMapping().getMappedIndex(ik));
    }

    /**
     * Marks what a value names for a store under a member the analysis cannot name, when the same
     * body also stores that value in the object's append contents: a list or set then holds no
     * element the analysis does not see, so only a list's positions are lost, while a dict's keys
     * still are (wala/ML#1178).
     *
     * @param pa The pointer analysis.
     * @param node The node of the store.
     * @param vn The object's value number.
     */
    void markMirrored(PointerAnalysis<InstanceKey> pa, CGNode node, int vn) {
      for (InstanceKey ik : pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, vn))) {
        TypeReference type = ik.concreteType().getReference();
        int index = pa.getInstanceKeyMapping().getMappedIndex(ik);
        if (type.equals(list)) {
          this.lists.add(index);
          this.unappended.add(index);
        }
        if (type.equals(dict)) this.open.add(index);
      }
    }

    /**
     * Marks every list and, when {@code adds}, every list, dict and set.
     *
     * @param adds Whether the mutation may also add an element the analysis does not record.
     */
    void markAll(boolean adds) {
      this.allLists = true;
      this.allUnappended = true;
      if (adds) this.allOpen = true;
    }
  }

  /**
   * Whether a type is a container the program can add an element to: a list, dict or set.
   *
   * @param type The type.
   * @return {@code true} iff it is such a container.
   */
  public static boolean isMutable(TypeReference type) {
    return type.equals(list) || type.equals(dict) || type.equals(set);
  }

  /**
   * Per builder, its {@link ContainerMutations}. They hold indices, not keys, so the cache pins
   * nothing the builder does not.
   */
  private static final Map<PropagationCallGraphBuilder, ContainerMutations> MUTATIONS =
      new WeakHashMap<>();

  /**
   * What the program may do to lists that the analysis does not record among their positioned
   * elements, so that their recorded positions need not be their run-time ones. Over every node of
   * the call graph, a list is so mutated when it is:
   *
   * <ul>
   *   <li>the receiver of a read of a repositioning method ({@link #REPOSITIONING_LIST_METHODS}),
   *       or an argument of a call of such a read, as in {@code list.append(l, x)};
   *   <li>the object of a store at a member that is not a constant index or name, such as {@code
   *       l[i] = x}. When the same body also stores the value in the object's append contents, as a
   *       comprehension's body does, the store adds no element the analysis does not see, so it
   *       opens only a dict;
   *   <li>the list of a slice assignment, {@code l[i:j] = m}, which the front end lowers to a call
   *       of the slice builtin on the list, as it does a slice read, but with the assigned value
   *       after the bounds ({@link #isSliceRead});
   *   <li>the left operand of a {@code +}, {@code *} or {@code |}, which {@code l += m}, {@code l
   *       *= n} and {@code s |= t} lower to and which mutate a list or set in place. The
   *       non-augmented forms build a new container, so this also takes their left operand, which
   *       only loses positions;
   *   <li>an argument of a call in program code that resolves to no target, such as one of {@code
   *       heapq.heappush}, {@code operator.iadd} or {@code getattr}, which the front end imports
   *       rather than summarizes, and so of the method {@code getattr(l, "append")} returns, unless
   *       the callee is a method read off a list (the repositioning methods govern those) or a
   *       function read off a TensorFlow or NumPy module or object ({@link
   *       #NON_RESIZING_LIBRARIES}), neither of which resizes a list.
   * </ul>
   *
   * A program function the call graph does not reach may still run, as a callback a library summary
   * does not call, such as a {@code key} of {@code sorted} or a function {@code map} applies. In
   * such a body, the receiver of a repositioning read is a list so mutated when it is a variable of
   * an enclosing frame or a global, read off what that variable is written. A receiver the body
   * builds itself is none the analysis has, and a library function that resizes nothing ({@link
   * #NON_RESIZING_LIBRARY_FUNCTIONS}) mutates nothing (wala/ML#1182). Any other receiver there may
   * be any list, and so is every list.
   *
   * <p>The front end drops the target of a {@code del} statement, so a subscript {@code del}
   * anywhere, in a reached node or such a body, may mutate any list.
   *
   * <p>Computed once per builder, once its call graph is built.
   *
   * @param builder The builder.
   * @return The mutations.
   */
  public static ContainerMutations of(PropagationCallGraphBuilder builder) {
    synchronized (MUTATIONS) {
      ContainerMutations cached = MUTATIONS.get(builder);
      if (cached != null) return cached;
    }
    ContainerMutations ret = compute(builder);
    remember(builder, ret);
    return ret;
  }

  /**
   * Records the mutations of a builder whose call graph is built, so {@link #of} answers with them
   * rather than scanning again.
   *
   * @param builder The builder.
   * @param mutations Its mutations, computed once its call graph was built.
   */
  static void remember(PropagationCallGraphBuilder builder, ContainerMutations mutations) {
    synchronized (MUTATIONS) {
      MUTATIONS.put(builder, mutations);
    }
  }

  /**
   * Whether the program may change a list's positioned elements unrecorded other than by a call of
   * {@code append} on it, which the analysis records in the list's appended contents as it solves;
   * see {@link #of}.
   *
   * @param builder The builder whose pointer analysis maps the list to its index.
   * @param list A list's instance key.
   * @return {@code true} iff its positioned elements and appended contents need not be its run-time
   *     elements.
   */
  boolean mayRepositionUnappended(PropagationCallGraphBuilder builder, InstanceKey list) {
    return allUnappended
        || unappended.contains(
            builder.getPointerAnalysis().getInstanceKeyMapping().getMappedIndex(list));
  }

  /**
   * Computes the mutations {@link #of} describes from the pointer analysis and the call graph as
   * they stand, without recording them: the builder computes them between solves, while the
   * analysis may still grow (wala/ML#1250).
   *
   * @param builder The builder.
   * @return The mutations.
   */
  static ContainerMutations compute(PropagationCallGraphBuilder builder) {
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    Marks marks = new Marks();
    Set<IMethod> reached = new HashSet<>();
    for (CGNode node : builder.getCallGraph()) {
      reached.add(node.getMethod());
      IR ir = node.getIR();
      if (ir == null) continue;
      SymbolTable symtab = ir.getSymbolTable();
      DefUse du = node.getDU();
      boolean program =
          node.getMethod() instanceof AstMethod && !node.getMethod().isWalaSynthetic();
      for (SSAInstruction instruction : ir.getInstructions()) {
        if (instruction instanceof AstPropertyRead read
            && (isRepositioning(symtab, read.getMemberRef())
                || isAdding(symtab, read.getMemberRef())))
          if (isAppendCall(symtab, du, read)) marks.markAppended(pa, node, read.getObjectRef());
          else marks.mark(pa, node, read.getObjectRef(), isAdding(symtab, read.getMemberRef()));
        else if (instruction instanceof AstPropertyWrite write
            && !symtab.isConstant(write.getMemberRef())) {
          // `d[k] = t` or `l[i] = t` with a key the analysis may not name, unless the body also
          // stores the value in the append contents, as a comprehension's body does (wala/ML#1178).
          if (storesInAppendContents(ir, symtab, write))
            marks.markMirrored(pa, node, write.getObjectRef());
          else marks.mark(pa, node, write.getObjectRef(), true);
        } else if (instruction instanceof SSABinaryOpInstruction binary
            && (binary.getOperator() == IBinaryOpInstruction.Operator.ADD
                || binary.getOperator() == IBinaryOpInstruction.Operator.MUL
                || binary.getOperator() == IBinaryOpInstruction.Operator.OR))
          // `l += m`, `l *= n` and `s |= t` mutate the left operand in place.
          marks.mark(pa, node, binary.getUse(0), true);
        else if (instruction instanceof SSAAbstractInvokeInstruction invoke) {
          Set<CGNode> targets =
              builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite());
          SSAInstruction callee = invoke.getNumberOfUses() > 0 ? du.getDef(invoke.getUse(0)) : null;
          if (calls(targets, DELETE_BUILTIN)) {
            if (invoke.getNumberOfUses() > 2) {
              if (!deletesByName(symtab, invoke))
                // A container the analysis does not see, such as one an unmodeled call returns, may
                // be any list.
                if (pa.getPointsToSet(
                        pa.getHeapModel().getPointerKeyForLocal(node, invoke.getUse(1)))
                    .isEmpty()) marks.markAll(false);
                else marks.mark(pa, node, invoke.getUse(1), false);
            } else if (deletesSubscript(node.getMethod(), invoke)) marks.markAll(false);
          } else if (calls(targets, SLICE_BUILTIN)) {
            // A slice, `l[i:j]`, read or assigned, is a call of the slice builtin on the list. A
            // read leaves the list as it is (wala/ML#1250).
            if (invoke.getNumberOfUses() > 1 && !isSliceRead(invoke))
              marks.mark(pa, node, invoke.getUse(1), true);
          } else if (callee instanceof AstPropertyRead method
              && isRepositioning(symtab, method.getMemberRef())) {
            // `list.append(l, x)`: the list is an argument, not the receiver.
            for (int i = 1; i < invoke.getNumberOfUses(); i++)
              marks.mark(pa, node, invoke.getUse(i), true);
          } else if (program && targets.isEmpty() && !isNonResizing(pa, node, callee))
            for (int i = 1; i < invoke.getNumberOfUses(); i++)
              marks.mark(pa, node, invoke.getUse(i), true);
        }
      }
    }
    scanUnreachedBodies(builder, reached, marks);
    return new ContainerMutations(
        marks.lists,
        marks.allLists,
        marks.unappended,
        marks.allUnappended,
        marks.open,
        marks.allOpen);
  }

  /**
   * Whether a read of a member is of {@code append}, and every use of what it reads calls it, as
   * {@code l.append(x)} does: the call the builder models by storing its argument in the list's
   * appended contents. Any other use, such as passing the method on, may call it where the builder
   * does not see it.
   *
   * @param symtab The symbol table of the read's method.
   * @param du The def-use of the read's method.
   * @param read The read.
   * @return {@code true} iff the read is of {@code append} and only called.
   */
  private static boolean isAppendCall(SymbolTable symtab, DefUse du, AstPropertyRead read) {
    int member = read.getMemberRef();
    if (!symtab.isStringConstant(member) || !"append".equals(symtab.getStringValue(member)))
      return false;
    if (du.getNumberOfUses(read.getDef()) == 0) return false;
    for (Iterator<SSAInstruction> uses = du.getUses(read.getDef()); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      if (!(use instanceof SSAAbstractInvokeInstruction invoke)
          || invoke.getNumberOfUses() == 0
          || invoke.getUse(0) != read.getDef()) return false;
      for (int i = 1; i < invoke.getNumberOfUses(); i++)
        if (invoke.getUse(i) == read.getDef()) return false;
    }
    return true;
  }

  /**
   * Whether a call of the slice builtin reads a slice of a list, {@code l[i:j:k]}, which leaves the
   * list as it is: the call passes the list and its three bounds, absent ones as {@code None}. A
   * slice assignment, {@code l[i:j] = m}, passes the assigned value after them, and a subscript by
   * a tuple, {@code x[i, j:k]}, which no list takes, passes a value per element of the tuple;
   * either may be taken for a mutation (wala/ML#1250).
   *
   * @param invoke A call of the slice builtin.
   * @return {@code true} iff it is a read of a slice of a list.
   */
  private static boolean isSliceRead(SSAAbstractInvokeInstruction invoke) {
    return invoke.getNumberOfUses() == 5;
  }

  /**
   * Scans the program functions the call graph does not reach; see {@link #of}.
   *
   * @param builder The builder.
   * @param reached The methods of the call graph's nodes.
   * @param marks The mutations, which this adds to.
   */
  private static void scanUnreachedBodies(
      PropagationCallGraphBuilder builder, Set<IMethod> reached, Marks marks) {
    // The values a variable is written, by its name and definer, which the receiver checks below
    // read for many reads of the same builtin or module: each is a scan of the whole call graph.
    Map<List<String>, Set<InstanceKey>> writes = new HashMap<>();
    for (IClass cls : builder.getClassHierarchy()) {
      Collection<? extends IMethod> methods = cls.getDeclaredMethods();
      if (methods == null) continue;
      for (IMethod method : methods) {
        if (!(method instanceof AstMethod) || method.isWalaSynthetic() || reached.contains(method))
          continue;
        IR ir;
        try {
          ir = builder.getAnalysisCache().getIR(method);
        } catch (RuntimeException e) {
          LOGGER.log(Level.FINE, e, () -> "Cannot read the IR of unreached " + method + ".");
          marks.markAll(true);
          return;
        }
        if (ir == null) continue;
        SymbolTable symtab = ir.getSymbolTable();
        DefUse du = builder.getAnalysisCache().getDefUse(ir);
        for (SSAInstruction instruction : ir.getInstructions()) {
          if (instruction instanceof SSAAbstractInvokeInstruction invoke
              && invoke.getNumberOfUses() > 0
              && readsName(du.getDef(invoke.getUse(0)), DELETE_NAME)) {
            if (invoke.getNumberOfUses() > 2) {
              if (!deletesByName(symtab, invoke)
                  && !isBuiltHere(builder, writes, du, du.getDef(invoke.getUse(1))))
                markReceiver(builder, du, invoke.getUse(1), marks, false);
            } else if (deletesSubscript(method, invoke)) marks.markAll(false);
          }
          if (!(instruction instanceof AstPropertyRead read)
              || !(isRepositioning(symtab, read.getMemberRef())
                  || isAdding(symtab, read.getMemberRef()))) continue;
          boolean adds = isAdding(symtab, read.getMemberRef());
          SSAInstruction receiver = du.getDef(read.getObjectRef());
          // A container the body builds itself is none the analysis allocates, and a library
          // function so named may resize none (wala/ML#1182).
          if (isBuiltHere(builder, writes, du, receiver)
              || isNonResizingLibraryFunction(
                  builder,
                  writes,
                  symtab,
                  du,
                  receiver,
                  symtab.getStringValue(read.getMemberRef()))) continue;
          markReceiver(builder, du, read.getObjectRef(), marks, adds);
        }
      }
    }
  }

  /**
   * Marks the containers a receiver in a body the call graph does not reach may be: those written
   * to the variable it reads, or every one when it reads none.
   *
   * @param builder The builder.
   * @param du The body's def-use.
   * @param vn The receiver.
   * @param marks The mutations, which this adds to.
   * @param adds Whether the mutation may also add an element the analysis does not record.
   */
  private static void markReceiver(
      PropagationCallGraphBuilder builder, DefUse du, int vn, Marks marks, boolean adds) {
    SSAInstruction receiver = du.getDef(vn);
    if (receiver instanceof AstLexicalRead lexical) {
      for (Access access : lexical.getAccesses())
        if (access.valueNumber() == vn)
          markLexicalWrites(builder, access.variableName(), access.variableDefiner(), marks, adds);
    } else if (receiver instanceof AstGlobalRead global)
      markGlobalWrites(builder, global.getGlobalName(), marks, adds);
    // A parameter, or a value the body computes: it may be any container.
    else marks.markAll(adds);
  }

  /**
   * Whether a receiver in a body the call graph does not reach is a container the body builds
   * itself: a display, such as {@code []} or {@code {}}, or a call of the {@code list}, {@code
   * dict} or {@code set} builtin, read by a name the call graph's nodes bind to no function of the
   * program's own. The analysis allocates no such container, so a mutation of it mutates none the
   * analysis has (wala/ML#1182). A rebinding of the name in a body the call graph does not reach is
   * not seen, which would take a {@code global} statement rebinding a builtin's name there.
   *
   * @param builder The builder.
   * @param writes The values variables are written, by name and definer, filled as read.
   * @param du The body's def-use.
   * @param receiver The instruction defining the receiver, or {@code null}.
   * @return {@code true} iff the body builds the receiver.
   */
  private static boolean isBuiltHere(
      PropagationCallGraphBuilder builder,
      Map<List<String>, Set<InstanceKey>> writes,
      DefUse du,
      SSAInstruction receiver) {
    if (receiver instanceof SSANewInstruction) return true;
    if (!(receiver instanceof SSAAbstractInvokeInstruction call) || call.getNumberOfUses() == 0)
      return false;
    SSAInstruction callee = du.getDef(call.getUse(0));
    String name;
    if (callee instanceof AstGlobalRead global) name = getBareName(global.getGlobalName());
    else if (callee instanceof AstLexicalRead lexical
        && lexical.getAccessCount() == 1
        && isModule(lexical.getAccess(0).variableDefiner()))
      name = lexical.getAccess(0).variableName();
    else return false;
    if (!CONTAINER_BUILTINS.contains(name)) return false;
    // A module may bind the name to a function of its own, which may return any container.
    for (InstanceKey value : getVariableWrites(builder, writes, name, null))
      if (value.concreteType().getName().toString().startsWith("Lscript ")) return false;
    return true;
  }

  /**
   * Whether a method read in a body the call graph does not reach is a library function that
   * resizes no container ({@link #NON_RESIZING_LIBRARY_FUNCTIONS}): its receiver is a chain of
   * attribute reads, as {@code tf.keras.backend} is, off a variable that every write in the call
   * graph binds to a TensorFlow or NumPy module ({@link #NON_RESIZING_LIBRARIES}), as {@code import
   * tensorflow as tf} does (wala/ML#1182).
   *
   * @param builder The builder.
   * @param writes The values variables are written, by name and definer, filled as read.
   * @param symtab The body's symbol table.
   * @param du The body's def-use.
   * @param receiver The instruction defining the receiver, or {@code null}.
   * @param method The method's name.
   * @return {@code true} iff the method is such a function.
   */
  private static boolean isNonResizingLibraryFunction(
      PropagationCallGraphBuilder builder,
      Map<List<String>, Set<InstanceKey>> writes,
      SymbolTable symtab,
      DefUse du,
      SSAInstruction receiver,
      String method) {
    // The attributes read off the root variable, outermost first.
    List<String> attributes = new ArrayList<>();
    SSAInstruction current = receiver;
    Set<SSAInstruction> seen = new HashSet<>();
    while (current instanceof AstPropertyRead read && seen.add(current)) {
      if (!symtab.isStringConstant(read.getMemberRef())) return false;
      attributes.add(0, symtab.getStringValue(read.getMemberRef()));
      current = du.getDef(read.getObjectRef());
    }
    String variable;
    String definer;
    if (current instanceof AstGlobalRead global) {
      variable = getBareName(global.getGlobalName());
      definer = null;
    } else if (current instanceof AstLexicalRead lexical && lexical.getAccessCount() == 1) {
      variable = lexical.getAccess(0).variableName();
      definer = lexical.getAccess(0).variableDefiner();
    } else return false;
    String path = (attributes.isEmpty() ? "" : "/" + String.join("/", attributes)) + "/" + method;
    // Only a read some listed function ends in, such as `update`, is worth the root's writes.
    if (NON_RESIZING_LIBRARY_FUNCTIONS.stream().noneMatch(function -> function.endsWith(path)))
      return false;
    boolean any = false;
    for (InstanceKey module : getVariableWrites(builder, writes, variable, definer)) {
      String type = module.concreteType().getName().toString();
      if (NON_RESIZING_LIBRARIES.stream().noneMatch(type::startsWith)
          || !NON_RESIZING_LIBRARY_FUNCTIONS.contains(type + path)) return false;
      any = true;
    }
    return any;
  }

  /**
   * The values the call graph's nodes write a variable: a global, or a module's variable, which is
   * how a module-level assignment is lowered, matched by bare name as {@link #markGlobalWrites}
   * matches them; or, given its definer, a variable of that frame.
   *
   * @param builder The builder.
   * @param writes The values variables are written, by name and definer, filled as read.
   * @param name The variable's bare name.
   * @param definer The name of the method whose frame defines the variable, or {@code null} for a
   *     global.
   * @return The written values' instance keys.
   */
  private static Set<InstanceKey> getVariableWrites(
      PropagationCallGraphBuilder builder,
      Map<List<String>, Set<InstanceKey>> writes,
      String name,
      String definer) {
    List<String> key = Arrays.asList(name, definer);
    Set<InstanceKey> cached = writes.get(key);
    if (cached != null) return cached;
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    Set<InstanceKey> ret = new HashSet<>();
    for (CGNode node : builder.getCallGraph()) {
      IR ir = node.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions())
        if (definer == null
            && instruction instanceof AstGlobalWrite write
            && name.equals(getBareName(write.getGlobalName())))
          pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, write.getVal()))
              .forEach(ret::add);
        else if (instruction instanceof AstLexicalWrite write)
          for (Access access : write.getAccesses())
            if (name.equals(access.variableName())
                && (definer == null
                    ? isModule(access.variableDefiner())
                    : definer.equals(access.variableDefiner())))
              pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, access.valueNumber()))
                  .forEach(ret::add);
    }
    writes.put(key, ret);
    return ret;
  }

  /**
   * Adds the lists a variable of a frame is written, in every node of the frame's method.
   *
   * @param builder The builder.
   * @param name The variable's name.
   * @param definer The name of the method whose frame defines the variable.
   * @param marks The mutations, which this adds to.
   * @param adds Whether the mutation may also add an element the analysis does not record.
   */
  private static void markLexicalWrites(
      PropagationCallGraphBuilder builder, String name, String definer, Marks marks, boolean adds) {
    for (CGNode node : builder.getCallGraph()) {
      IR ir = node.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions())
        if (instruction instanceof AstLexicalWrite write)
          for (Access access : write.getAccesses())
            if (name.equals(access.variableName()) && definer.equals(access.variableDefiner()))
              marks.mark(builder.getPointerAnalysis(), node, access.valueNumber(), adds);
    }
  }

  /**
   * Adds the lists a global is written, in every node: by a write of the global, or by a module's
   * write of its variable of that name, which is how a module-level assignment is lowered. The
   * global is matched by its bare name, so a same-named variable of another module counts as well,
   * which only loses positions.
   *
   * @param builder The builder.
   * @param name The global's name, as its read names it.
   * @param marks The mutations, which this adds to.
   * @param adds Whether the mutation may also add an element the analysis does not record.
   */
  private static void markGlobalWrites(
      PropagationCallGraphBuilder builder, String name, Marks marks, boolean adds) {
    String bare = getBareName(name);
    for (CGNode node : builder.getCallGraph()) {
      IR ir = node.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions())
        if (instruction instanceof AstGlobalWrite write) {
          if (bare.equals(getBareName(write.getGlobalName())))
            marks.mark(builder.getPointerAnalysis(), node, write.getVal(), adds);
        } else if (instruction instanceof AstLexicalWrite write)
          for (Access access : write.getAccesses())
            if (bare.equals(access.variableName()) && isModule(access.variableDefiner()))
              marks.mark(builder.getPointerAnalysis(), node, access.valueNumber(), adds);
    }
  }

  /**
   * The bare name of a global: what follows its last space or slash, as in {@code global H} or
   * {@code global script m.py/H}.
   *
   * @param name The global's name.
   * @return Its bare name.
   */
  private static String getBareName(String name) {
    if (name == null) return "";
    return name.substring(Math.max(name.lastIndexOf(' '), name.lastIndexOf('/')) + 1);
  }

  /**
   * Whether a frame's name is a module's: a script's, whose name ends in the file's, rather than a
   * function's within it.
   *
   * @param definer The frame's name.
   * @return {@code true} iff it names a module.
   */
  private static boolean isModule(String definer) {
    return definer != null && definer.startsWith("Lscript ") && definer.endsWith(".py");
  }

  /**
   * Whether an instruction reads a variable of a name: a global or a variable of an enclosing
   * frame.
   *
   * @param instruction The instruction, or {@code null}.
   * @param name The name.
   * @return {@code true} iff it reads that name.
   */
  private static boolean readsName(SSAInstruction instruction, String name) {
    if (instruction instanceof AstGlobalRead global)
      return global.getGlobalName() != null && global.getGlobalName().endsWith(name);
    if (instruction instanceof AstLexicalRead lexical)
      for (Access access : lexical.getAccesses())
        if (name.equals(access.variableName())) return true;
    if (instruction instanceof SSANewInstruction allocation)
      return allocation.getConcreteType().getName().toString().endsWith(name);
    return false;
  }

  /**
   * Whether a body stores the value of a store under a member the analysis cannot name also in the
   * same object's append contents ({@link
   * PythonSSAPropagationCallGraphBuilder#LIST_APPEND_CONTENTS_FIELD}), where the element readers
   * look. A comprehension's body stores each element so (wala/ML#1178).
   *
   * @param ir The body.
   * @param symtab Its symbol table.
   * @param write The store.
   * @return {@code true} iff the body has such a second store.
   */
  private static boolean storesInAppendContents(IR ir, SymbolTable symtab, AstPropertyWrite write) {
    for (SSAInstruction other : ir.getInstructions())
      if (other instanceof AstPropertyWrite store
          && store.getObjectRef() == write.getObjectRef()
          && store.getValue() == write.getValue()
          && symtab.isStringConstant(store.getMemberRef())
          && PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD.equals(
              symtab.getStringValue(store.getMemberRef()))) return true;
    return false;
  }

  /**
   * Whether a member is the name of a repositioning method ({@link #REPOSITIONING_LIST_METHODS}).
   *
   * @param symtab The symbol table.
   * @param member The member's value number.
   * @return {@code true} iff it is such a name.
   */
  private static boolean isRepositioning(SymbolTable symtab, int member) {
    return symtab.isStringConstant(member)
        && REPOSITIONING_LIST_METHODS.contains(symtab.getStringValue(member));
  }

  /**
   * Whether a member is the name of a method that may add an element unrecorded ({@link
   * #ADDING_METHODS}).
   *
   * @param symtab The symbol table.
   * @param member The member's value number.
   * @return {@code true} iff it is such a name.
   */
  private static boolean isAdding(SymbolTable symtab, int member) {
    return symtab.isStringConstant(member)
        && ADDING_METHODS.contains(symtab.getStringValue(member));
  }

  /**
   * Whether an unresolved call's callee cannot resize a list passed to it: a method read off a
   * list, which the repositioning methods govern, or a function read off a TensorFlow or NumPy
   * module or object ({@link #NON_RESIZING_LIBRARIES}), directly or through attributes, as {@code
   * tf.image.random_crop} is read off {@code tf}.
   *
   * @param pa The pointer analysis.
   * @param node The calling node.
   * @param callee The instruction defining the callee, or {@code null}.
   * @return {@code true} iff the callee cannot resize a list.
   */
  private static boolean isNonResizing(
      PointerAnalysis<InstanceKey> pa, CGNode node, SSAInstruction callee) {
    // The owner may be an attribute of a module, `tf.image`, which the analysis allocates as a
    // plain
    // object or not at all: read up the chain of attribute reads for a library's allocation.
    DefUse du = node.getDU();
    Set<SSAInstruction> seen = new HashSet<>();
    for (SSAInstruction current = callee;
        current instanceof AstPropertyRead read && seen.add(current);
        current = du.getDef(read.getObjectRef())) {
      OrdinalSet<InstanceKey> owners =
          pa.getPointsToSet(pa.getHeapModel().getPointerKeyForLocal(node, read.getObjectRef()));
      for (InstanceKey owner : owners) {
        if (current == callee && owner.concreteType().getReference().equals(list)) return true;
        String type = owner.concreteType().getName().toString();
        for (String library : NON_RESIZING_LIBRARIES) if (type.startsWith(library)) return true;
      }
    }
    return false;
  }

  /**
   * Whether a call of the deletion builtin deletes by a name: an attribute, {@code del o.a}, or a
   * string key, {@code del d["k"]}, neither of which resizes a list. The front end passes a
   * subscript's or attribute's {@code del} what it deletes from and the key or name (wala/ML#1239).
   *
   * @param symtab The symbol table of the call's method.
   * @param invoke A call of the deletion builtin passing what it deletes from and a key.
   * @return {@code true} iff the key is a string.
   */
  private static boolean deletesByName(SymbolTable symtab, SSAAbstractInvokeInstruction invoke) {
    return symtab.isStringConstant(invoke.getUse(2));
  }

  /**
   * Whether a call of the deletion builtin that passes no container is a subscript {@code del}, as
   * in {@code del l[0]}, read from the statement's source text, for a front end that drops the
   * target. A statement whose text cannot be read is taken to be one.
   *
   * @param method The method holding the call.
   * @param invoke The call.
   * @return {@code true} unless the statement's text is read and has no subscript.
   */
  private static boolean deletesSubscript(IMethod method, SSAAbstractInvokeInstruction invoke) {
    if (!(method instanceof AstMethod astMethod) || invoke.iIndex() < 0) return true;
    Position at = astMethod.getSourcePosition(invoke.iIndex());
    if (at == null || at.getFirstLine() <= 0) return true;
    try (BufferedReader reader = new BufferedReader(at.getReader())) {
      String text =
          reader
              .lines()
              .skip(at.getFirstLine() - 1)
              .limit(Math.max(1, at.getLastLine() - at.getFirstLine() + 1))
              .collect(Collectors.joining("\n"));
      return text.indexOf('[') >= 0;
    } catch (IOException | RuntimeException e) {
      LOGGER.log(Level.FINE, e, () -> "Cannot read the source of a del in " + method + ".");
      return true;
    }
  }

  /**
   * Whether a call's targets include a builtin.
   *
   * @param targets The call's targets.
   * @param builtin The builtin's type name, such as {@link #DELETE_BUILTIN}.
   * @return {@code true} iff a target is that builtin.
   */
  private static boolean calls(Set<CGNode> targets, String builtin) {
    for (CGNode target : targets)
      if (target.getMethod().getDeclaringClass().getName().toString().equals(builtin)) return true;
    return false;
  }
}
