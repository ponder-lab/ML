package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code tf.stack(values, axis=0, name='stack')}. Computes the precise output shape
 * by combining the {@code values} list's PTS-derived length {@code N} with the first element's
 * shape: {@code values[0].shape[:axis] + (N,) + values[0].shape[axis:]}. The output dtype is
 * inherited from the first element of {@code values}.
 *
 * <p>Currently handles {@code axis=0} (default) and constant {@code axis} values that resolve to
 * {@code 0} or a positive index in range. For non-zero {@code axis}, the implementation reads the
 * constant axis value if available; non-constant axis falls back to ⊤ shape.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/stack">tf.stack</a>
 * @see <a href="https://github.com/wala/ML/issues/449">wala/ML#449</a> (Tier 5).
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Stack extends TensorGenerator {

  @SuppressWarnings("unused")
  private static final Logger LOGGER = getLogger(Stack.class.getName());

  /**
   * Parameter positions and keyword names for {@code tf.stack(values, axis=0, name='stack')}.
   * Ordinals match the position in {@code tensorflow.xml}'s {@code paramNames} after the implicit
   * {@code self} receiver, so {@code Parameters.VALUES.getIndex() == 0} resolves to the first
   * user-facing positional argument.
   */
  protected enum Parameters {
    /** The list of tensors to stack. */
    VALUES,

    /** The axis along which to insert the new dimension. Default {@code 0}. */
    AXIS,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in argument-resolution helpers.
     *
     * @return The lowercased enum name (e.g. {@code "values"}).
     */
    public String getName() {
      return name().toLowerCase();
    }

    /**
     * Positional index of this parameter, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based positional index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public Stack(PointsToSetVariable source) {
    super(source);
  }

  public Stack(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> valuesPts =
        this.getArgumentPointsToSet(
            builder, Parameters.VALUES.getIndex(), Parameters.VALUES.getName());
    if (valuesPts == null || valuesPts.isEmpty()) return null;

    Integer axis = resolveConstantAxis(builder);
    if (axis == null) return null; // axis present but unresolved → ⊤.

    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    for (InstanceKey valIk : valuesPts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(valIk);
      if (asin == null) continue;
      TypeReference ref = asin.getConcreteType().getReference();
      if (!(ref.equals(list) || ref.equals(tuple))) continue;

      OrdinalSet<InstanceKey> catalog =
          pa.getPointsToSet(
              ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                  .getPointerKeyForObjectCatalog(asin));
      int n = catalog.size();
      if (n == 0) continue;

      // Read first element's shape via field "0" of the values list.
      Set<List<Dimension<?>>> firstShapes = getShapesOfFirstElement(builder, asin, catalog);
      if (firstShapes == null) continue;

      for (List<Dimension<?>> firstShape : firstShapes) {
        // Negative-axis normalization: TF allows axis ∈ [-(R+1), R+1] where R = rank of first
        // element. After normalization the axis is the insertion index in [0, R].
        int rank = firstShape.size();
        int normalizedAxis = axis < 0 ? axis + rank + 1 : axis;
        if (normalizedAxis < 0 || normalizedAxis > rank) continue; // invalid → skip
        List<Dimension<?>> outShape = new ArrayList<>(rank + 1);
        outShape.addAll(firstShape.subList(0, normalizedAxis));
        outShape.add(new NumericDim(n));
        outShape.addAll(firstShape.subList(normalizedAxis, rank));
        ret.add(outShape);
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> valuesPts =
        this.getArgumentPointsToSet(
            builder, Parameters.VALUES.getIndex(), Parameters.VALUES.getName());
    if (valuesPts == null || valuesPts.isEmpty()) return EnumSet.of(DType.UNKNOWN);

    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    Set<DType> ret = EnumSet.noneOf(DType.class);

    for (InstanceKey valIk : valuesPts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(valIk);
      if (asin == null) continue;
      TypeReference ref = asin.getConcreteType().getReference();
      if (!(ref.equals(list) || ref.equals(tuple))) continue;

      OrdinalSet<InstanceKey> catalog =
          pa.getPointsToSet(
              ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                  .getPointerKeyForObjectCatalog(asin));
      OrdinalSet<InstanceKey> firstElemPts = getFirstElementPts(builder, asin, catalog);
      if (firstElemPts == null) continue;
      Set<DType> firstDTypes = this.getDTypesOfValue(builder, firstElemPts);
      if (firstDTypes != null) ret.addAll(firstDTypes);
    }
    return ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret;
  }

  /**
   * Reads the {@code axis} argument's PTS and resolves it to a concrete integer if all keys are
   * {@code ConstantKey<Number>}. Returns {@code 0} when {@code axis} isn't passed (TF default) and
   * {@code null} when the argument is present but unresolvable (caller should emit ⊤).
   */
  private Integer resolveConstantAxis(PropagationCallGraphBuilder builder) {
    boolean axisPresent =
        this.isKeywordArgumentPresent(builder, Parameters.AXIS.getName())
            || this.getNumberOfPossiblePositionalArguments(builder).stream()
                .anyMatch(n -> n > Parameters.AXIS.getIndex());
    if (!axisPresent) return 0;
    OrdinalSet<InstanceKey> axisPts =
        this.getArgumentPointsToSet(builder, Parameters.AXIS.getIndex(), Parameters.AXIS.getName());
    if (axisPts == null || axisPts.isEmpty()) return null;
    Integer ret = null;
    for (InstanceKey ik : axisPts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object val = ((ConstantKey<?>) ik).getValue();
      if (!(val instanceof Number)) return null;
      int v = ((Number) val).intValue();
      if (ret == null) ret = v;
      else if (ret != v) return null; // ambiguous axis → ⊤
    }
    return ret == null ? 0 : ret;
  }

  /**
   * Reads field {@code "0"} of the {@code values} list/tuple and returns the shapes of that
   * element, or {@code null} if the field can't be resolved.
   */
  private Set<List<Dimension<?>>> getShapesOfFirstElement(
      PropagationCallGraphBuilder builder,
      AllocationSiteInNode listAsin,
      OrdinalSet<InstanceKey> catalog) {
    OrdinalSet<InstanceKey> firstElemPts = getFirstElementPts(builder, listAsin, catalog);
    if (firstElemPts == null) return null;
    return this.getShapesOfValue(builder, firstElemPts);
  }

  /**
   * Resolves the points-to set of field {@code "0"} on the given list/tuple alloc, or {@code null}
   * if the field can't be resolved.
   */
  private OrdinalSet<InstanceKey> getFirstElementPts(
      PropagationCallGraphBuilder builder,
      AllocationSiteInNode listAsin,
      OrdinalSet<InstanceKey> catalog) {
    for (InstanceKey catalogIK : catalog) {
      if (!(catalogIK instanceof ConstantKey)) continue;
      Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
      if (fieldIndex == null || fieldIndex != 0) continue;
      FieldReference subscript =
          FieldReference.findOrCreate(Root, findOrCreateAsciiAtom("0"), Root);
      IField f = builder.getClassHierarchy().resolveField(subscript);
      if (f == null) continue;
      PointerKey pk = builder.getPointerKeyForInstanceField(listAsin, f);
      return builder.getPointerAnalysis().getPointsToSet(pk);
    }
    return null;
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
