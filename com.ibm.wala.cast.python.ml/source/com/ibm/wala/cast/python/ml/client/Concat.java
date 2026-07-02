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
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code tf.concat(values, axis=0, name='concat')}. Computes the precise output shape
 * by walking every entry in the {@code values} list, summing each input's dim along the resolved
 * {@code axis}, and inheriting the rest of the shape from the first input. The output dtype is
 * inherited from the first element of {@code values}.
 *
 * <p>Falls back to ⊤ shape when:
 *
 * <ul>
 *   <li>The {@code values} argument's PTS doesn't resolve to a list/tuple.
 *   <li>The {@code axis} argument is non-constant or evaluates to multiple values.
 *   <li>Any input element's shape can't be resolved or has a non-numeric dim at the {@code axis}
 *       position (e.g., a {@code SymbolicDim}).
 * </ul>
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/concat">tf.concat</a>
 * @see <a href="https://github.com/wala/ML/issues/449">wala/ML#449</a> (Tier 5).
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Concat extends TensorGenerator {

  @SuppressWarnings("unused")
  private static final Logger LOGGER = getLogger(Concat.class.getName());

  /**
   * Parameter positions and keyword names for {@code tf.concat(values, axis=0, name='concat')}.
   * Ordinals match the position in {@code tensorflow.xml}'s {@code paramNames} after the implicit
   * {@code self} receiver, so {@code Parameters.VALUES.getIndex() == 0} resolves to the first
   * user-facing positional argument.
   */
  protected enum Parameters {
    /**
     * The list of tensors to concatenate; all must have the same rank and matching non-axis dims.
     */
    VALUES,

    /** The axis along which to concatenate. Default {@code 0}. */
    AXIS,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in argument-resolution helpers.
     *
     * @return The lowercased enum name (e.g. {@code "values"}).
     */
    public String getName() {
      return name().toLowerCase(Locale.ROOT);
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

  public Concat(PointsToSetVariable source) {
    super(source);
  }

  public Concat(CGNode node) {
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
      TypeReference ref = asin.concreteType().getReference();
      if (!(ref.equals(list) || ref.equals(tuple))) continue;

      OrdinalSet<InstanceKey> catalog =
          pa.getPointsToSet(
              ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                  .getPointerKeyForObjectCatalog(asin));
      if (catalog.size() == 0) continue;

      List<Dimension<?>> outShape = computeConcatenatedShape(builder, asin, catalog, axis);
      if (outShape != null) ret.add(outShape);
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
      TypeReference ref = asin.concreteType().getReference();
      if (!(ref.equals(list) || ref.equals(tuple))) continue;

      OrdinalSet<InstanceKey> catalog =
          pa.getPointsToSet(
              ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                  .getPointerKeyForObjectCatalog(asin));
      OrdinalSet<InstanceKey> firstElemPts = getElementPts(builder, asin, catalog, 0);
      if (firstElemPts == null) continue;
      Set<DType> firstDTypes = this.getDTypesOfValue(builder, firstElemPts);
      if (firstDTypes != null) ret.addAll(firstDTypes);
    }
    return ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret;
  }

  /**
   * Computes the concatenated shape: takes the first input's shape verbatim except at the {@code
   * axis} position, where the dim is replaced by the sum across all inputs. Returns {@code null} if
   * any input's shape can't be resolved or has a non-{@link NumericDim} at the axis.
   */
  private List<Dimension<?>> computeConcatenatedShape(
      PropagationCallGraphBuilder builder,
      AllocationSiteInNode listAsin,
      OrdinalSet<InstanceKey> catalog,
      int axis) {
    // Get the first element's shape as the template.
    OrdinalSet<InstanceKey> firstElemPts = getElementPts(builder, listAsin, catalog, 0);
    if (firstElemPts == null) return null;
    Set<List<Dimension<?>>> firstShapes = this.getShapesOfValue(builder, firstElemPts);
    if (firstShapes == null || firstShapes.isEmpty()) return null;
    // Pick any one; in practice we expect a single shape for the list.
    List<Dimension<?>> firstShape = firstShapes.iterator().next();
    int rank = firstShape.size();
    int normalizedAxis = axis < 0 ? axis + rank : axis;
    if (normalizedAxis < 0 || normalizedAxis >= rank) return null;

    // Sum the axis dim across all elements.
    long sum = 0;
    for (InstanceKey catalogIK : catalog) {
      if (!(catalogIK instanceof ConstantKey)) return null;
      Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
      if (fieldIndex == null) return null;
      OrdinalSet<InstanceKey> elemPts = getElementPts(builder, listAsin, catalog, fieldIndex);
      if (elemPts == null) return null;
      Set<List<Dimension<?>>> elemShapes = this.getShapesOfValue(builder, elemPts);
      if (elemShapes == null || elemShapes.isEmpty()) return null;
      List<Dimension<?>> elemShape = elemShapes.iterator().next();
      if (elemShape.size() != rank) return null; // rank mismatch — can't concat soundly
      Dimension<?> axisDim = elemShape.get(normalizedAxis);
      if (!(axisDim instanceof NumericDim)) return null; // non-numeric → can't sum
      sum += ((NumericDim) axisDim).value();
    }

    List<Dimension<?>> outShape = new ArrayList<>(firstShape);
    outShape.set(normalizedAxis, new NumericDim((int) sum));
    return outShape;
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
      else if (ret != v) return null;
    }
    return ret == null ? 0 : ret;
  }

  /**
   * Resolves the points-to set of the element at {@code fieldIndex} on the given list/tuple alloc,
   * or {@code null} if the field can't be resolved.
   */
  private OrdinalSet<InstanceKey> getElementPts(
      PropagationCallGraphBuilder builder,
      AllocationSiteInNode listAsin,
      OrdinalSet<InstanceKey> catalog,
      int fieldIndex) {
    for (InstanceKey catalogIK : catalog) {
      if (!(catalogIK instanceof ConstantKey)) continue;
      Integer idx = getFieldIndex((ConstantKey<?>) catalogIK);
      if (idx == null || idx != fieldIndex) continue;
      FieldReference subscript =
          FieldReference.findOrCreate(
              Root, findOrCreateAsciiAtom(Integer.toString(fieldIndex)), Root);
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
