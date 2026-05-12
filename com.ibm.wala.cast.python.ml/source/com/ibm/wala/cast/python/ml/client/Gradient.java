package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.GradientTape.gradient}. Returns a fresh tensor whose shape and dtype
 * match the {@code sources} argument—the gradient of a function w.r.t. a tensor has the same shape
 * and dtype as that tensor. The output is a distinct allocation from {@code sources} (no input
 * alias).
 *
 * <p>When {@code sources} is a list or tuple (the common Keras pattern {@code tape.gradient(loss,
 * model.trainable_variables)}), the runtime returns a parallel list of fresh tensors—one per
 * source. The {@link TupleElementProvider} implementation here lets subscript reads like {@code
 * gradients[i]} resolve to the shape/dtype of the i-th source. The aggregate {@code
 * getDefaultShapes}/{@code getDefaultDTypes} path still recovers a single shape/dtype for the
 * single-source case via {@code getShapesOfValue}/{@code getDTypesOfValue}'s existing aggregation
 * behavior.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/GradientTape#gradient">tf.GradientTape.gradient</a>
 * @see <a href="https://github.com/wala/ML/issues/430">wala/ML#430</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Gradient extends TensorGenerator implements TupleElementProvider {

  private enum Parameters {
    TARGET,
    SOURCES,
    OUTPUT_GRADIENTS,
    UNCONNECTED_GRADIENTS;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Gradient(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code Gradient} anchored to a manual node. Used by {@link
   * TensorGenerator#createManualGenerator} so that the synthetic-{@code do()} fallback path can
   * recover shape/dtype for gradient-produced tensors without a caller-side {@link
   * PointsToSetVariable}.
   *
   * @param node The {@link CGNode} for the {@code gradient.do()} synthetic method.
   */
  public Gradient(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return shapesOfArg(builder, Parameters.SOURCES.getIndex(), Parameters.SOURCES.getName());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> dtypes =
        dtypesOfArg(builder, Parameters.SOURCES.getIndex(), Parameters.SOURCES.getName());
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  private Set<List<Dimension<?>>> shapesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, pts);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }
    return this.getArgumentShapesViaCallers(builder, paramPos, paramName);
  }

  private Set<DType> dtypesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, pts);
      if (dtypes != null && !dtypes.isEmpty()) return dtypes;
    }
    return this.getArgumentDTypesViaCallers(builder, paramPos, paramName);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Returns {@code true} when {@code sources}'s points-to set contains a list or tuple
   *     allocation, indicating the gradient result should be modeled as a per-source list/tuple.
   */
  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> sourcesPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.SOURCES.getIndex(), Parameters.SOURCES.getName());
    if (sourcesPTS != null && !sourcesPTS.isEmpty()) {
      for (InstanceKey ik : sourcesPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null) {
          TypeReference ref = asin.getConcreteType().getReference();
          if (ref.equals(list) || ref.equals(tuple)) return true;
        }
      }
    }
    return false;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Walks {@code sources}'s list/tuple allocation and returns the shape of the element at
   *     the requested index. If the per-index field's PTS is empty, returns {@code null} (lattice
   *     ⊤) rather than falling through to the aggregate, which would silently leak sibling indices'
   *     shapes (see <a href="https://github.com/wala/ML/issues/396">wala/ML#396</a>).
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> sourcesPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.SOURCES.getIndex(), Parameters.SOURCES.getName());
    if (sourcesPTS != null && !sourcesPTS.isEmpty()) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (InstanceKey ik : sourcesPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin == null) continue;
        TypeReference ref = asin.getConcreteType().getReference();
        if (!ref.equals(list) && !ref.equals(tuple)) continue;

        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            builder
                .getPointerAnalysis()
                .getPointsToSet(
                    ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                        .getPointerKeyForObjectCatalog(asin));

        for (InstanceKey catalogIK : objectCatalogPointsToSet) {
          if (!(catalogIK instanceof ConstantKey)) continue;
          Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
          if (fieldIndex == null || fieldIndex != index) continue;

          FieldReference subscript =
              FieldReference.findOrCreate(
                  asin.getConcreteType().getReference(),
                  findOrCreateAsciiAtom(fieldIndex.toString()),
                  Root);
          IField f = builder.getClassHierarchy().resolveField(subscript);
          if (f == null) continue;

          PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
          Set<List<Dimension<?>>> fieldShapes =
              this.getShapesOfValue(builder, builder.getPointerAnalysis().getPointsToSet(pk));
          if (fieldShapes == null || fieldShapes.isEmpty()) return null;
          ret.addAll(fieldShapes);
        }
      }
      if (!ret.isEmpty()) return ret;
      return null;
    }
    return this.getShapes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Walks {@code sources}'s list/tuple allocation and returns the dtype of the element at
   *     the requested index. If the per-index field's PTS is empty, returns {@code UNKNOWN}
   *     (lattice ⊤) rather than falling through to the aggregate (see <a
   *     href="https://github.com/wala/ML/issues/396">wala/ML#396</a>).
   */
  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> sourcesPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.SOURCES.getIndex(), Parameters.SOURCES.getName());
    if (sourcesPTS != null && !sourcesPTS.isEmpty()) {
      Set<DType> ret = HashSetFactory.make();
      for (InstanceKey ik : sourcesPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin == null) continue;
        TypeReference ref = asin.getConcreteType().getReference();
        if (!ref.equals(list) && !ref.equals(tuple)) continue;

        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            builder
                .getPointerAnalysis()
                .getPointsToSet(
                    ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                        .getPointerKeyForObjectCatalog(asin));

        for (InstanceKey catalogIK : objectCatalogPointsToSet) {
          if (!(catalogIK instanceof ConstantKey)) continue;
          Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
          if (fieldIndex == null || fieldIndex != index) continue;

          FieldReference subscript =
              FieldReference.findOrCreate(
                  asin.getConcreteType().getReference(),
                  findOrCreateAsciiAtom(fieldIndex.toString()),
                  Root);
          IField f = builder.getClassHierarchy().resolveField(subscript);
          if (f == null) continue;

          PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
          Set<DType> fieldDTypes =
              this.getDTypesOfValue(builder, builder.getPointerAnalysis().getPointsToSet(pk));
          if (fieldDTypes == null || fieldDTypes.isEmpty()) return EnumSet.of(DType.UNKNOWN);
          ret.addAll(fieldDTypes);
        }
      }
      if (!ret.isEmpty()) return ret;
      return EnumSet.of(DType.UNKNOWN);
    }
    return this.getDTypes(builder);
  }

  /** {@inheritDoc} */
  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> shapes = this.getShapesForIndex(builder, index);
    Set<DType> dTypes = this.getDTypesForIndex(builder, index);

    Set<TensorType> ret = HashSetFactory.make();

    if (shapes == null) {
      for (DType dtype : dTypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), null));
      return ret;
    }

    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), dimensionList));

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
