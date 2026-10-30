package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.dict;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.set;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.parseIntOrNull;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstGlobalWrite;
import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.debug.UnimplementedError;
import com.ibm.wala.util.intset.IntSet;
import com.ibm.wala.util.intset.IntSetUtil;
import com.ibm.wala.util.intset.MutableIntSet;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeSet;
import java.util.WeakHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * The elements of a container instance, as the analysis models them: the client-facing authority on
 * a container's length, the key of each element, and whether an element may be {@code None}.
 *
 * <p>A client asks this rather than reading the container's object catalog, the names of the
 * analysis's internal element channels ({@link
 * PythonSSAPropagationCallGraphBuilder#LIST_APPEND_CONTENTS_FIELD}, {@link
 * PythonSSAPropagationCallGraphBuilder#LIST_OPERATION_CONTENTS_FIELD}) or the encoding of element
 * indices, all of which are the analysis's own conventions and may change with it.
 *
 * <p>A container here means a container in the analysis's element model: a list, tuple, set, dict
 * or {@code enumerate} object, the same set {@code PythonTypes.isContainer} names. It does not mean
 * {@code collections.abc.Container}: an {@code enumerate} object, for one, is a one-pass iterator
 * with no length, exhausted after one pass. Any other instance has no element keys here beyond
 * those its catalog records.
 *
 * <p>The length is reported only when it is known: a {@link
 * com.ibm.wala.cast.python.types.PythonTypes#list} or {@link
 * com.ibm.wala.cast.python.types.PythonTypes#tuple} display that program code allocates, whose
 * elements all sit at constant indices forming the run {@code 0..n-1}, with no element whose
 * position is unknown (one appended, built by a comprehension, unpacked from a starred element, or
 * produced by list concatenation or repetition), and, for a list, that nothing in the program may
 * mutate unrecorded (see {@link #mutations}). A list or tuple a library summary or an operator
 * model allocates reports none: its catalog holds the model's placeholder slots, not its run-time
 * elements. Every other container reports no length.
 *
 * <p>A client queries this after the analysis, on the builder the analysis ran on. Those queries
 * don't keep the builder reachable through the generator memos, which only an analysis in progress
 * writes (<a href="https://github.com/wala/ML/issues/1170">wala/ML#1170</a>). A client done with a
 * builder may still call {@link TensorGenerator#clearCaches} to release any other per-builder memo
 * deterministically.
 */
public final class ContainerElements {

  private static final Logger LOGGER = Logger.getLogger(ContainerElements.class.getName());

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
   * them, so an unresolved call of one marks nothing; see {@link #mutations}.
   */
  private static final List<String> NON_RESIZING_LIBRARIES = List.of("Ltensorflow", "Lnumpy");

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

  /**
   * What the program may do to containers that the analysis does not record.
   *
   * @param lists The instance keys, by their index in the pointer analysis's instance key mapping,
   *     of the lists whose positioned elements it may so change.
   * @param allLists Whether it may so change any list's at all.
   * @param open The instance keys, by the same index, of the lists, dicts and sets it may add an
   *     element to under a key the analysis does not record.
   * @param allOpen Whether it may so add an element to any list, dict or set at all.
   */
  private record Mutations(IntSet lists, boolean allLists, IntSet open, boolean allOpen) {}

  /** The mutations {@link #mutations} is collecting. */
  private static final class Marks {

    /** See {@link Mutations#lists}. */
    final MutableIntSet lists = IntSetUtil.make();

    /** See {@link Mutations#open}. */
    final MutableIntSet open = IntSetUtil.make();

    /** See {@link Mutations#allLists}. */
    boolean allLists;

    /** See {@link Mutations#allOpen}. */
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
        if (type.equals(list)) this.lists.add(index);
        if (adds && isMutable(type)) this.open.add(index);
      }
    }

    /**
     * Marks every list and, when {@code adds}, every list, dict and set.
     *
     * @param adds Whether the mutation may also add an element the analysis does not record.
     */
    void markAll(boolean adds) {
      this.allLists = true;
      if (adds) this.allOpen = true;
    }
  }

  /**
   * Whether a type is a container the program can add an element to: a list, dict or set.
   *
   * @param type The type.
   * @return {@code true} iff it is such a container.
   */
  private static boolean isMutable(TypeReference type) {
    return type.equals(list) || type.equals(dict) || type.equals(set);
  }

  /**
   * Per builder, its {@link Mutations}. They hold indices, not keys, so the cache pins nothing the
   * builder does not.
   */
  private static final Map<PropagationCallGraphBuilder, Mutations> MUTATIONS = new WeakHashMap<>();

  /** The length, or -1 when it is not known. */
  private final int arity;

  /** The element key at each position, in order; empty when the length is not known. */
  private final List<PointerKey> positions;

  /** Every key that can hold an element. */
  private final Set<PointerKey> allElements;

  /** Whether {@link #allElements} holds every key that can hold an element. */
  private final boolean complete;

  /** The builder whose pointer analysis holds the keys. */
  private final PropagationCallGraphBuilder builder;

  /**
   * How a value's elements are read when they are not read off {@link #allElements} and {@link
   * #positions}: a {@link #ofValue value} that is a structured component, or one of several
   * containers. {@code null} for an instance's elements.
   */
  private final Reader reader;

  /** A reading of elements that are not the keys of one container instance. */
  private interface Reader {

    List<Set<TensorType>> positionTypes(TensorTypeAnalysis analysis);

    Set<TensorType> elementTypes(TensorTypeAnalysis analysis);

    boolean mayHoldNone();
  }

  private ContainerElements(
      int arity,
      List<PointerKey> positions,
      Set<PointerKey> allElements,
      boolean complete,
      PropagationCallGraphBuilder builder) {
    this(arity, positions, allElements, complete, builder, null);
  }

  private ContainerElements(
      int arity,
      List<PointerKey> positions,
      Set<PointerKey> allElements,
      boolean complete,
      PropagationCallGraphBuilder builder,
      Reader reader) {
    this.arity = arity;
    this.positions = Collections.unmodifiableList(positions);
    this.allElements = Collections.unmodifiableSet(allElements);
    this.complete = complete;
    this.builder = builder;
    this.reader = reader;
  }

  /**
   * A dataset element's component, the value a generator reads at a path of selectors within the
   * element a {@link TupleElementProvider} generates: the inputs a Keras step hands to {@code call}
   * when {@code fit} is given a dataset, or the components a loop over a dataset unpacks.
   *
   * @param provider The element's provider.
   * @param path The selectors, outermost first.
   * @param generator The generator reading the component.
   */
  public record DatasetComponent(
      TupleElementProvider provider, List<Object> path, TensorGenerator generator) {

    /**
     * The length of the tuple this component is, when the program declares that tuple ({@link
     * TupleElementProvider#declaredArity}).
     *
     * @param builder The builder.
     * @return The length, or empty when the component is no declared tuple.
     */
    public OptionalInt declaredArity(PropagationCallGraphBuilder builder) {
      return this.provider.declaredArity(builder, this.path);
    }
  }

  /**
   * The dataset element's component a value is, when the factory's generator for it reads one: a
   * {@link DatasetTupleElementGenerator} over a {@link TupleElementProvider}.
   *
   * @param builder The builder.
   * @param value The value.
   * @return The component, or {@code null} when the value is no such component.
   */
  public static DatasetComponent datasetComponent(
      PropagationCallGraphBuilder builder, PointerKey value) {
    if (builder.getPropagationSystem().isImplicit(value)) return null;
    TensorGenerator generator;
    try {
      generator =
          TensorGeneratorFactory.getGenerator(
              builder.getPropagationSystem().findOrCreatePointsToSet(value), builder);
    } catch (IllegalArgumentException | UnimplementedError e) {
      return null;
    }
    return generator instanceof DatasetTupleElementGenerator element
            && element.getUnderlying() instanceof TupleElementProvider provider
        ? new DatasetComponent(provider, element.getPath(), element)
        : null;
  }

  /**
   * Reads the elements of a value, whatever the instances it may hold. The value's points-to set
   * need not name a container: a dataset element's component, such as the {@code inputs} a Keras
   * step hands to {@code call} when {@code fit} is given {@code from_tensor_slices(((x, mask),
   * y))}, is a tuple at run time that no allocation in the pointer analysis stands for.
   *
   * <ul>
   *   <li>A value that is a dataset element's component ({@link #datasetComponent}) the program
   *       declares as a tuple ({@link TupleElementProvider#declaredArity}) has that tuple's length.
   *       Its {@link #positionTypes} are the types the element's provider gives each declared
   *       position, read only below that length; no flow-graph variable holds a position, so they
   *       are the provider's reading rather than the dataflow's. Its {@link #positions()} are
   *       empty, as it has no element keys. It is {@linkplain #isComplete() complete} only when
   *       each position is known to be a tensor: its types are read, none is wholly unknown, and
   *       the provider does not take it for a structure ({@link
   *       TupleElementProvider#isStructureAt}).
   *   <li>A value that is a component the program does not declare as a tuple, such as a tensor, or
   *       a tuple on one branch and a tensor on another, has no length. Its {@link #elementTypes}
   *       are the component's own types. It is complete only when it is known to be a tensor, as a
   *       declared position is: no types, a wholly unknown one, which a component that may be a
   *       structure carries, or a structure the provider reports leave it incomplete.
   *   <li>Any other value merges the {@link #of containers} its points-to set holds: their common
   *       length, when they agree, the union of their types, and completeness only when each is
   *       complete. Its {@link #positions()} are those of its one container, and empty for several.
   *       A value whose points-to set is empty has no length and is not complete.
   * </ul>
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose pointer analysis models the value.
   * @param value The value.
   * @return Its elements.
   */
  public static ContainerElements ofValue(PropagationCallGraphBuilder builder, PointerKey value) {
    DatasetComponent component = datasetComponent(builder, value);
    if (component != null) {
      OptionalInt declared = component.declaredArity(builder);
      if (declared.isPresent()) {
        List<Set<TensorType>> positions = new ArrayList<>(declared.getAsInt());
        boolean complete = true;
        for (int i = 0; i < declared.getAsInt(); i++) {
          List<Object> at = new ArrayList<>(component.path());
          at.add(i);
          Set<TensorType> types = pathTypes(builder, component.provider(), at);
          positions.add(types == null ? null : Collections.unmodifiableSet(new HashSet<>(types)));
          complete &= isTensor(builder, component.provider(), at, types);
        }
        return new ContainerElements(
            declared.getAsInt(), List.of(), Set.of(), complete, builder, structure(positions));
      }
      Set<TensorType> types = componentTypes(builder, component);
      return new ContainerElements(
          -1,
          List.of(),
          Set.of(),
          isTensor(builder, component.provider(), component.path(), types),
          builder,
          flat(types));
    }
    List<ContainerElements> parts = new ArrayList<>();
    for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(value))
      parts.add(of(builder, ik));
    if (parts.size() == 1) return parts.get(0);
    if (parts.isEmpty())
      return new ContainerElements(-1, List.of(), Set.of(), false, builder, flat(Set.of()));
    int arity = parts.get(0).arity;
    boolean complete = true;
    Set<PointerKey> allElements = new LinkedHashSet<>();
    for (ContainerElements part : parts) {
      if (part.arity != arity) arity = -1;
      complete &= part.complete;
      allElements.addAll(part.allElements);
    }
    return new ContainerElements(
        arity, List.of(), allElements, complete, builder, merged(parts, arity));
  }

  /**
   * Whether a component is known to be a tensor of the types read for it: the types are read, none
   * is wholly unknown, and the provider does not take the component for a structure. Any other
   * component may be a structure, whose members its types do not show, or may be anything.
   *
   * @param provider The element's provider.
   * @param path The component's selectors.
   * @param types The types read for the component, {@code null} when unreadable.
   * @return {@code true} iff the component is known to be a tensor of those types.
   */
  private static boolean isTensor(
      PropagationCallGraphBuilder builder,
      TupleElementProvider provider,
      List<Object> path,
      Set<TensorType> types) {
    if (types == null || types.isEmpty() || types.stream().anyMatch(ContainerElements::isTop))
      return false;
    try {
      return !provider.isStructureAt(builder, path);
    } catch (IllegalArgumentException | UnimplementedError e) {
      return false;
    }
  }

  /**
   * A provider's types at a path.
   *
   * @return The types, or {@code null} when the provider cannot read them.
   */
  private static Set<TensorType> pathTypes(
      PropagationCallGraphBuilder builder, TupleElementProvider provider, List<Object> path) {
    try {
      return provider.getTensorTypesForPath(builder, path);
    } catch (IllegalArgumentException | UnimplementedError e) {
      return null;
    }
  }

  /** Whether a type is wholly unknown: no known shape and no known dtype. */
  private static boolean isTop(TensorType type) {
    return type.getDims() == null && type.getDType() == DType.UNKNOWN;
  }

  /**
   * A component's own tensor types, as its provider reads them. A single index is read with {@link
   * TupleElementProvider#getTensorTypesForIndex}, which marks a component that is a tensor on some
   * of the joined datasets and a structure on others with a wholly unknown member.
   *
   * @return The types, or {@code null} when the provider cannot read them.
   */
  private static Set<TensorType> componentTypes(
      PropagationCallGraphBuilder builder, DatasetComponent component) {
    List<Object> path = component.path();
    if (!(path.size() == 1 && path.get(0) instanceof Integer index))
      return pathTypes(builder, component.provider(), path);
    try {
      return component.provider().getTensorTypesForIndex(builder, index);
    } catch (IllegalArgumentException | UnimplementedError e) {
      return null;
    }
  }

  /**
   * The reading of a declared tuple component: its provider's types at each declared index.
   *
   * @param positions Per declared index, the types read there, {@code null} when unreadable.
   */
  private static Reader structure(List<Set<TensorType>> positions) {
    return new Reader() {
      @Override
      public List<Set<TensorType>> positionTypes(TensorTypeAnalysis analysis) {
        List<Set<TensorType>> ret = new ArrayList<>();
        for (Set<TensorType> types : positions) ret.add(types == null ? Set.of() : types);
        return Collections.unmodifiableList(ret);
      }

      @Override
      public Set<TensorType> elementTypes(TensorTypeAnalysis analysis) {
        Set<TensorType> ret = new HashSet<>();
        for (Set<TensorType> types : this.positionTypes(analysis)) ret.addAll(types);
        return Collections.unmodifiableSet(ret);
      }

      @Override
      public boolean mayHoldNone() {
        // A dataset's component may be None (tf.data's NoneTensorSpec), which its type shows; a
        // position the provider cannot read, or reads as wholly unknown, may be anything.
        for (Set<TensorType> types : positions)
          if (types == null || types.stream().anyMatch(t -> t.isNone() || isTop(t))) return true;
        return false;
      }
    };
  }

  /** The reading of a component that is no declared tuple: its own types, and no positions. */
  private static Reader flat(Set<TensorType> types) {
    return new Reader() {
      @Override
      public List<Set<TensorType>> positionTypes(TensorTypeAnalysis analysis) {
        return List.of();
      }

      @Override
      public Set<TensorType> elementTypes(TensorTypeAnalysis analysis) {
        return types == null ? Set.of() : Collections.unmodifiableSet(new HashSet<>(types));
      }

      @Override
      public boolean mayHoldNone() {
        return types == null || types.stream().anyMatch(t -> t.isNone() || isTop(t));
      }
    };
  }

  /** The reading of several containers: per position and in all, the union of theirs. */
  private static Reader merged(List<ContainerElements> parts, int arity) {
    return new Reader() {
      @Override
      public List<Set<TensorType>> positionTypes(TensorTypeAnalysis analysis) {
        if (arity < 0) return List.of();
        List<Set<TensorType>> ret = new ArrayList<>(arity);
        for (int i = 0; i < arity; i++) ret.add(new HashSet<>());
        for (ContainerElements part : parts) {
          List<Set<TensorType>> types = part.positionTypes(analysis);
          for (int i = 0; i < arity && i < types.size(); i++) ret.get(i).addAll(types.get(i));
        }
        List<Set<TensorType>> unmodifiable = new ArrayList<>(arity);
        for (Set<TensorType> types : ret) unmodifiable.add(Collections.unmodifiableSet(types));
        return Collections.unmodifiableList(unmodifiable);
      }

      @Override
      public Set<TensorType> elementTypes(TensorTypeAnalysis analysis) {
        Set<TensorType> ret = new HashSet<>();
        for (ContainerElements part : parts) ret.addAll(part.elementTypes(analysis));
        return Collections.unmodifiableSet(ret);
      }

      @Override
      public boolean mayHoldNone() {
        for (ContainerElements part : parts) if (part.mayHoldNone()) return true;
        return false;
      }
    };
  }

  /**
   * Reads the elements of a container instance.
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose pointer analysis models the
   *     container.
   * @param container The container instance.
   * @return Its elements.
   */
  public static ContainerElements of(PropagationCallGraphBuilder builder, InstanceKey container) {
    OrdinalSet<InstanceKey> catalog =
        builder
            .getPointerAnalysis()
            .getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(container));
    TypeReference type = container.concreteType().getReference();
    boolean keyed = type.equals(dict);

    TreeSet<Integer> indices = new TreeSet<>();
    Set<String> fieldNames = new LinkedHashSet<>();
    boolean unpositioned = false;
    boolean complete = true;

    if (catalog != null)
      for (InstanceKey name : catalog) {
        if (!(name instanceof ConstantKey<?> constant)) {
          // A key the analysis could not name: an element may sit anywhere, under any key.
          unpositioned = true;
          complete = false;
          continue;
        }
        Object value = constant.getValue();
        if (PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD.equals(value)
            || PythonSSAPropagationCallGraphBuilder.LIST_OPERATION_CONTENTS_FIELD.equals(value)) {
          unpositioned = true;
          fieldNames.add((String) value);
          continue;
        }
        Integer index = index(value);
        if (index != null) {
          indices.add(index);
          fieldNames.add(index.toString());
        } else if (keyed)
          // Any constant keys a dict: a float, a bool, None or a string, the field named by its
          // text. An attribute field of the dict counts too, which only adds a key.
          fieldNames.add(String.valueOf(value));
        // Otherwise an attribute name the catalog also records, such as a method read off the
        // instance, which is no element (wala/ML#603).
      }

    // Only a display program code allocates has its run-time elements in its catalog.
    if (!isProgramDisplay(container)) unpositioned = true;

    // A container the program may mutate in a way the analysis does not record in its elements.
    if (isMutable(type)) {
      Mutations mutations = mutations(builder);
      int index = mappedIndex(builder, container);
      if (type.equals(list) && (mutations.allLists() || mutations.lists().contains(index)))
        unpositioned = true;
      if (mutations.allOpen() || mutations.open().contains(index)) complete = false;
    }

    // The channels that hold elements whose position is unknown are element keys of every
    // positional or set-like container, whether or not its catalog names them: a list built by
    // concatenation or repetition keeps its elements there without recording the name.
    if (!keyed) {
      fieldNames.add(PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD);
      fieldNames.add(PythonSSAPropagationCallGraphBuilder.LIST_OPERATION_CONTENTS_FIELD);
    }

    Set<PointerKey> allElements = new LinkedHashSet<>();
    for (String fieldName : fieldNames) {
      PointerKey key = fieldKey(builder, container, fieldName);
      if (key != null) allElements.add(key);
    }

    boolean sequence = type.equals(list) || type.equals(tuple);
    boolean contiguous =
        indices.isEmpty() || (indices.first() == 0 && indices.last() == indices.size() - 1);
    if (!sequence || unpositioned || !contiguous)
      return new ContainerElements(-1, List.of(), allElements, complete, builder);

    List<PointerKey> positions = new ArrayList<>(indices.size());
    for (int i = 0; i < indices.size(); i++) {
      PointerKey key = fieldKey(builder, container, Integer.toString(i));
      // An index the class hierarchy cannot resolve to a field leaves its element unreadable, and
      // with it the length's correspondence to the keys.
      if (key == null) return new ContainerElements(-1, List.of(), allElements, complete, builder);
      positions.add(key);
    }
    return new ContainerElements(indices.size(), positions, allElements, complete, builder);
  }

  /**
   * The container's length, when the analysis knows it.
   *
   * @return The number of elements, or empty when the length is not known.
   */
  public OptionalInt arity() {
    return this.arity < 0 ? OptionalInt.empty() : OptionalInt.of(this.arity);
  }

  /**
   * The key of the element at each position.
   *
   * @return The keys of positions {@code 0..n-1}, where {@code n} is {@link #arity()}; empty when
   *     the length is not known, and for a {@linkplain #ofValue value} that is a dataset element's
   *     component or may be one of several containers, whose positions have no single key.
   */
  public List<PointerKey> positions() {
    return this.positions;
  }

  /**
   * Whether {@link #allElements()} holds every key that can hold an element of the container. It
   * does not when the analysis saw an element stored under a key it could not name, such as {@code
   * d[k] = t} with {@code k} unresolved, or the program may add one by a means it does not record,
   * such as {@code l.extend(m)} or an unresolved call the container is passed to. The element keys,
   * and the types read off them, are then a subset of the run-time ones.
   *
   * @return {@code true} iff the element keys are all of them.
   */
  public boolean isComplete() {
    return this.complete;
  }

  /**
   * Every key that can hold an element of the container, including those of elements whose position
   * is not known. Exact only when {@link #isComplete()}; otherwise a subset.
   *
   * @return The element keys.
   */
  public Set<PointerKey> allElements() {
    return this.allElements;
  }

  /**
   * The tensor types the analysis gives the element at each position.
   *
   * @param analysis The solved {@link TensorTypeAnalysis}.
   * @return Per position of {@link #positions()}, the element's tensor types, an empty set when the
   *     analysis gives it none; empty when the length is not known. Exact only when {@link
   *     #isComplete()}.
   */
  public List<Set<TensorType>> positionTypes(TensorTypeAnalysis analysis) {
    if (this.reader != null) return this.reader.positionTypes(analysis);
    List<Set<TensorType>> ret = new ArrayList<>(this.positions.size());
    for (PointerKey key : this.positions) {
      Set<TensorType> types = new HashSet<>();
      this.addTypes(analysis, key, types);
      ret.add(Collections.unmodifiableSet(types));
    }
    return Collections.unmodifiableList(ret);
  }

  /**
   * The tensor types the analysis gives the container's elements, whatever their positions: the
   * element types of a container of unknown or known length, over every element key ({@link
   * #allElements()}), those of elements whose position is unknown included. Each key is read as
   * {@link #positionTypes} reads one.
   *
   * @param analysis The solved {@link TensorTypeAnalysis}.
   * @return The union of the elements' tensor types; empty when no element has tensor evidence.
   *     Exact only when {@link #isComplete()}; otherwise the types of the elements the keys name,
   *     and the client chooses the conservative reading for its own question.
   */
  public Set<TensorType> elementTypes(TensorTypeAnalysis analysis) {
    if (this.reader != null) return this.reader.elementTypes(analysis);
    Set<TensorType> ret = new HashSet<>();
    for (PointerKey key : this.allElements) this.addTypes(analysis, key, ret);
    return Collections.unmodifiableSet(ret);
  }

  /**
   * Adds the tensor types the analysis gives a key.
   *
   * @param analysis The solved analysis.
   * @param key The key.
   * @param types The types, which this adds to.
   */
  private void addTypes(TensorTypeAnalysis analysis, PointerKey key, Set<TensorType> types) {
    PointsToSetVariable variable = this.variable(key);
    // A key the analysis's flow graph does not hold has no state, rather than an empty one.
    if (variable == null || !analysis.getProblem().getFlowGraph().containsNode(variable)) return;
    Set<TensorType> state = analysis.getOutState(variable);
    if (state != null) types.addAll(state);
  }

  /**
   * Whether an element of the container may be {@code None}, over every element key ({@link
   * #allElements()}), read with {@link TensorGenerator#anyNullConstant(OrdinalSet)}. When the keys
   * are not all of them ({@link #isComplete()}), an element may be anything, {@code None} too.
   *
   * @return {@code true} iff some element key may hold the {@code None} constant, or the keys are
   *     not all of them.
   */
  public boolean mayHoldNone() {
    if (!this.complete) return true;
    if (this.reader != null) return this.reader.mayHoldNone();
    for (PointerKey key : this.allElements)
      if (TensorGenerator.anyNullConstant(this.builder.getPointerAnalysis().getPointsToSet(key)))
        return true;
    return false;
  }

  /**
   * The element index a catalog key names, whatever its encoding.
   *
   * @param value The catalog key's value.
   * @return The index, or {@code null} when the key names no index.
   */
  private static Integer index(Object value) {
    if (value instanceof Integer i) return i >= 0 ? i : null;
    if (value instanceof Long l)
      return l >= 0 && l <= Integer.MAX_VALUE ? Integer.valueOf(l.intValue()) : null;
    if (value instanceof String s) {
      Integer i = parseIntOrNull(s);
      return i != null && i >= 0 ? i : null;
    }
    return null;
  }

  /**
   * The key of a container's field.
   *
   * @param builder The builder.
   * @param container The container.
   * @param name The field's name.
   * @return The key, or {@code null} when the class hierarchy resolves no such field.
   */
  private static PointerKey fieldKey(
      PropagationCallGraphBuilder builder, InstanceKey container, String name) {
    IField field =
        builder
            .getClassHierarchy()
            .resolveField(FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(name), Root));
    return field == null ? null : builder.getPointerKeyForInstanceField(container, field);
  }

  /**
   * Whether a container is a list or tuple display that program code allocates: an allocation site
   * in a node whose method is program code ({@link AstMethod}, not a summary), at a {@code new} of
   * that node's IR. An operator model's allocation keyed at another instruction, such as a call or
   * a binary operation, is none.
   *
   * @param container The container.
   * @return {@code true} iff the container is such a display.
   */
  private static boolean isProgramDisplay(InstanceKey container) {
    if (!(container instanceof AllocationSiteInNode asin)) return false;
    CGNode node = asin.getNode();
    if (!(node.getMethod() instanceof AstMethod) || node.getMethod().isWalaSynthetic())
      return false;
    IR ir = node.getIR();
    if (ir == null) return false;
    for (SSAInstruction instruction : ir.getInstructions())
      if (instruction instanceof SSANewInstruction allocation
          && allocation.getNewSite().equals(asin.getSite())) return true;
    return false;
  }

  /**
   * What the program may do to lists that the analysis does not record among their positioned
   * elements, so that their recorded positions need not be their run-time ones. Over every node of
   * the call graph, a list is so mutated when it is:
   *
   * <ul>
   *   <li>the receiver of a read of a repositioning method ({@link #REPOSITIONING_LIST_METHODS}),
   *       or an argument of a call of such a read, as in {@code list.append(l, x)};
   *   <li>the object of a store at a member that is not a constant index or name, such as {@code
   *       l[i] = x};
   *   <li>the list of a slice, read or assigned, which the front end lowers to a call of the slice
   *       builtin on the list;
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
   * an enclosing frame or a global, read off what that variable is written. Any other receiver
   * there may be any list, and so is every list.
   *
   * <p>The front end drops the target of a {@code del} statement, so a subscript {@code del}
   * anywhere, in a reached node or such a body, may mutate any list.
   *
   * <p>Computed once per builder.
   *
   * @param builder The builder.
   * @return The mutations.
   */
  private static Mutations mutations(PropagationCallGraphBuilder builder) {
    synchronized (MUTATIONS) {
      Mutations cached = MUTATIONS.get(builder);
      if (cached != null) return cached;
    }
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
          marks.mark(pa, node, read.getObjectRef(), isAdding(symtab, read.getMemberRef()));
        else if (instruction instanceof AstPropertyWrite write
            && !symtab.isConstant(write.getMemberRef()))
          // `d[k] = t` or `l[i] = t` with a key the analysis may not name.
          marks.mark(pa, node, write.getObjectRef(), true);
        else if (instruction instanceof SSABinaryOpInstruction binary
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
            if (deletesSubscript(node.getMethod(), invoke)) marks.markAll(false);
          } else if (calls(targets, SLICE_BUILTIN)) {
            // A slice, `l[i:j]`, read or assigned, is a call of the slice builtin on the list.
            if (invoke.getNumberOfUses() > 1) marks.mark(pa, node, invoke.getUse(1), true);
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
    Mutations ret = new Mutations(marks.lists, marks.allLists, marks.open, marks.allOpen);
    synchronized (MUTATIONS) {
      MUTATIONS.put(builder, ret);
    }
    return ret;
  }

  /**
   * Scans the program functions the call graph does not reach; see {@link #mutations}.
   *
   * @param builder The builder.
   * @param reached The methods of the call graph's nodes.
   * @param marks The mutations, which this adds to.
   */
  private static void scanUnreachedBodies(
      PropagationCallGraphBuilder builder, Set<IMethod> reached, Marks marks) {
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
              && readsName(du.getDef(invoke.getUse(0)), DELETE_NAME)
              && deletesSubscript(method, invoke)) marks.markAll(false);
          if (!(instruction instanceof AstPropertyRead read)
              || !(isRepositioning(symtab, read.getMemberRef())
                  || isAdding(symtab, read.getMemberRef()))) continue;
          boolean adds = isAdding(symtab, read.getMemberRef());
          SSAInstruction receiver = du.getDef(read.getObjectRef());
          if (receiver instanceof AstLexicalRead lexical) {
            for (Access access : lexical.getAccesses())
              if (access.valueNumber() == read.getObjectRef())
                markLexicalWrites(
                    builder, access.variableName(), access.variableDefiner(), marks, adds);
          } else if (receiver instanceof AstGlobalRead global)
            markGlobalWrites(builder, global.getGlobalName(), marks, adds);
          // A parameter, or a value the body computes: it may be any container.
          else marks.markAll(adds);
        }
      }
    }
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
    String bare = bareName(name);
    for (CGNode node : builder.getCallGraph()) {
      IR ir = node.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions())
        if (instruction instanceof AstGlobalWrite write) {
          if (bare.equals(bareName(write.getGlobalName())))
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
  private static String bareName(String name) {
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
   * Whether a call of the deletion builtin is a subscript {@code del}, as in {@code del l[0]}, read
   * from the statement's source text, since the front end drops its target. A statement whose text
   * cannot be read is taken to be one.
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

  /**
   * A container's index in the pointer analysis's instance key mapping.
   *
   * @param builder The builder.
   * @param container The container.
   * @return The index.
   */
  private static int mappedIndex(PropagationCallGraphBuilder builder, InstanceKey container) {
    return builder.getPointerAnalysis().getInstanceKeyMapping().getMappedIndex(container);
  }

  /**
   * The flow-graph variable of a key, when the pointer analysis holds it explicitly.
   *
   * @param key The key.
   * @return The variable, or {@code null} for an implicitly represented key.
   */
  private PointsToSetVariable variable(PointerKey key) {
    if (this.builder.getPropagationSystem().isImplicit(key)) return null;
    try {
      return this.builder.getPropagationSystem().findOrCreatePointsToSet(key);
    } catch (UnimplementedError e) {
      return null;
    }
  }
}
