package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.dict;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.parseIntOrNull;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder.ElementMirrorKey;
import com.ibm.wala.cast.python.util.ContainerMutations;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeSet;

/**
 * The elements of a container instance, as the analysis models them: a container's length, the key
 * of each element, and whether an element may be {@code None}, read off the container's object
 * catalog and the element channels ({@link
 * PythonSSAPropagationCallGraphBuilder#LIST_APPEND_CONTENTS_FIELD}, {@link
 * PythonSSAPropagationCallGraphBuilder#LIST_OPERATION_CONTENTS_FIELD}). It is internal to this
 * package, for the analysis's own reads of a container's length and elements; a client queries a
 * container through {@link Containers}.
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
 * mutate unrecorded (see {@link ContainerMutations#of}). A list or tuple a library summary or an
 * operator model allocates reports none: its catalog holds the model's placeholder slots, not its
 * run-time elements. Every other container reports no length.
 */
final class ContainerElements {

  /** The length, or -1 when it is not known. */
  private final int arity;

  /** The element key at each position, in order; empty when the length is not known. */
  private final List<PointerKey> positions;

  /** Every key that can hold an element. */
  private final Set<PointerKey> allElements;

  /** Whether {@link #getAllElements} holds every key that can hold an element. */
  private final boolean complete;

  /** The builder whose pointer analysis holds the keys. */
  private final PropagationCallGraphBuilder builder;

  private ContainerElements(
      int arity,
      List<PointerKey> positions,
      Set<PointerKey> allElements,
      boolean complete,
      PropagationCallGraphBuilder builder) {
    this.arity = arity;
    this.positions = Collections.unmodifiableList(positions);
    this.allElements = Collections.unmodifiableSet(allElements);
    this.complete = complete;
    this.builder = builder;
  }

  /**
   * Reads the elements of a container instance.
   *
   * @param builder The {@link PropagationCallGraphBuilder} whose pointer analysis models the
   *     container.
   * @param container The container instance.
   * @return Its elements.
   */
  static ContainerElements of(PropagationCallGraphBuilder builder, InstanceKey container) {
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
        Integer index = getIndex(value);
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
    if (ContainerMutations.isMutable(type)) {
      ContainerMutations mutations = ContainerMutations.of(builder);
      int index = getMappedIndex(builder, container);
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
      PointerKey key = getFieldKey(builder, container, fieldName);
      if (key != null) allElements.add(key);
    }

    boolean sequence = type.equals(list) || type.equals(tuple);
    boolean contiguous =
        indices.isEmpty() || (indices.first() == 0 && indices.last() == indices.size() - 1);
    if (!sequence || unpositioned || !contiguous)
      return new ContainerElements(-1, List.of(), allElements, complete, builder);

    List<PointerKey> positions = new ArrayList<>(indices.size());
    for (int i = 0; i < indices.size(); i++) {
      PointerKey key = getFieldKey(builder, container, Integer.toString(i));
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
  OptionalInt getArity() {
    return this.arity < 0 ? OptionalInt.empty() : OptionalInt.of(this.arity);
  }

  /**
   * The key of the element at each position.
   *
   * @return The keys of positions {@code 0..n-1}, where {@code n} is {@link #getArity()}; empty
   *     when the length is not known.
   */
  List<PointerKey> getPositions() {
    return this.positions;
  }

  /**
   * Whether {@link #getAllElements()} holds every key that can hold an element of the container. It
   * does not when the analysis saw an element stored under a key it could not name, such as {@code
   * d[k] = t} with {@code k} unresolved, or the program may add one by a means it does not record,
   * such as {@code l.extend(m)} or an unresolved call the container is passed to. The element keys,
   * and the types read off them, are then a subset of the run-time ones.
   *
   * @return {@code true} iff the element keys are all of them.
   */
  boolean isComplete() {
    return this.complete;
  }

  /**
   * Every key that can hold an element of the container, including those of elements whose position
   * is not known. Exact only when {@link #isComplete()}; otherwise a subset.
   *
   * @return The element keys.
   */
  Set<PointerKey> getAllElements() {
    return this.allElements;
  }

  /**
   * Whether an element of the container may be {@code None}, over every element key ({@link
   * #getAllElements()}), read with {@link TensorGenerator#anyNullConstant(OrdinalSet)}. When the
   * keys are not all of them ({@link #isComplete()}), an element may be anything, {@code None} too.
   *
   * @return {@code true} iff some element key may hold the {@code None} constant, or the keys are
   *     not all of them.
   */
  boolean mayHoldNone() {
    if (!this.complete) return true;
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
  private static Integer getIndex(Object value) {
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
  private static PointerKey getFieldKey(
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
    // A dataset element's structure copies the positions of the structure it was built from
    // (wala/ML#1190), so its catalog holds run-time elements exactly when its source's does.
    if (container instanceof ElementMirrorKey mirror && !mirror.isLeaf())
      return isProgramDisplay(mirror.getSource());
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
   * A container's index in the pointer analysis's instance key mapping.
   *
   * @param builder The builder.
   * @param container The container.
   * @return The index.
   */
  private static int getMappedIndex(PropagationCallGraphBuilder builder, InstanceKey container) {
    return builder.getPointerAnalysis().getInstanceKeyMapping().getMappedIndex(container);
  }
}
