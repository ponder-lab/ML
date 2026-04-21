package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by {@code tf.data.Dataset.from_tensor_slices}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetFromTensorSlicesGenerator extends DatasetGenerator
    implements TupleElementProvider {

  private static final Logger LOGGER =
      Logger.getLogger(DatasetFromTensorSlicesGenerator.class.getName());

  protected enum Parameters {
    TENSORS,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public DatasetFromTensorSlicesGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetFromTensorSlicesGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // For tf.data.Dataset.from_tensor_slices(tensors), the dataset elements are created by
    // slicing the input tensors along their first dimension. Thus, the element shapes are
    // the input shapes with the first dimension removed.
    // The 'tensors' argument is at position 0 (args: this, tensors, name).
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    Set<List<Dimension<?>>> inputShapes = null;
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      inputShapes = this.getShapesOfTensorsArgument(builder, tensorsPTS);
    }
    final int tensorsPTSSize = tensorsPTS == null ? -1 : tensorsPTS.size();
    final Set<List<Dimension<?>>> ptsPathShapes = inputShapes;
    LOGGER.fine(
        () ->
            "DatasetFromTensorSlicesGenerator.getDefaultShapes: source="
                + this.getSource()
                + ", tensorsPTS size="
                + tensorsPTSSize
                + ", inputShapes via pts-path="
                + (ptsPathShapes == null ? "null" : ptsPathShapes.size() + " shapes"));

    // Fallback: if the points-to set for the argument is empty (e.g., `tensors` is the result of
    // a Python binary op, for which WALA does not allocate a trackable target), walk the call
    // string to resolve the argument value number in each caller and delegate to getShapes, which
    // knows how to construct an ElementWiseOperation generator for binop-def'd locals.
    if (inputShapes == null || inputShapes.isEmpty()) {
      inputShapes =
          this.getArgumentShapesViaCallers(
              builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
      final Set<List<Dimension<?>>> fallbackShapes = inputShapes;
      LOGGER.fine(
          () ->
              "DatasetFromTensorSlicesGenerator.getDefaultShapes: fallback inputShapes="
                  + (fallbackShapes == null ? "null" : fallbackShapes.size() + " shapes"));
    }

    if (inputShapes == null) return null;
    if (inputShapes.isEmpty()) return Collections.emptySet();

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes) {
      if (shape.size() > 0) {
        // Remove the first dimension to account for slicing.
        ret.add(new ArrayList<>(shape.subList(1, shape.size())));
      } else {
        // If the input is already a scalar (unexpected for from_tensor_slices),
        // the element shape is empty.
        ret.add(Collections.emptyList());
      }
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // For from_tensor_slices, element dtypes are the same as the input tensor(s)' dtypes.
    // The 'tensors' argument is at position 0 (args: this, tensors, name).
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfTensorsArgument(builder, tensorsPTS);
      if (!dtypes.isEmpty()) return dtypes;
    }

    // Fallback: walk the call string to resolve the argument in each caller.
    Set<DType> fallback =
        this.getArgumentDTypesViaCallers(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    if (fallback != null && !fallback.isEmpty()) return fallback;

    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    Set<List<Dimension<?>>> inputShapes = null;
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      inputShapes = this.getShapesOfTensorsArgument(builder, tensorsPTS);
    }

    // Fallback mirroring `getDefaultShapes`: when the `tensors` argument has empty PTS (e.g.,
    // it's the SSA result of a binop chain feeding `from_tensor_slices`), walk callers to
    // resolve its shape. Without this, partial-batch size computation in
    // `DatasetBatchGenerator` degrades to a symbolic `?` even though the dataset's size is
    // statically knowable. See wala/ML#357.
    if (inputShapes == null || inputShapes.isEmpty()) {
      inputShapes =
          this.getArgumentShapesViaCallers(
              builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    }

    if (inputShapes == null) return Collections.emptySet();

    Set<Long> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes) {
      if (!shape.isEmpty()) {
        Dimension<?> firstDim = shape.get(0);
        if (firstDim instanceof NumericDim) {
          ret.add(Long.valueOf(((NumericDim) firstDim).value()));
        }
      }
    }
    return ret;
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && asin.getConcreteType().getReference().equals(tuple)) {
          return true;
        }
      }
    }
    return false;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && asin.getConcreteType().getReference().equals(tuple)) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null && fieldIndex == index) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      asin.getConcreteType().getReference(),
                      findOrCreateAsciiAtom(fieldIndex.toString()),
                      Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                Set<List<Dimension<?>>> fieldShapes =
                    this.getShapesOfValue(builder, builder.getPointerAnalysis().getPointsToSet(pk));
                if (fieldShapes != null)
                  for (List<Dimension<?>> shape : fieldShapes) {
                    if (shape.size() > 0) {
                      ret.add(new ArrayList<>(shape.subList(1, shape.size())));
                    } else {
                      ret.add(Collections.emptyList());
                    }
                  }
              }
            }
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
      // Per-index lookup saw a tuple but the requested field's PTS was empty — return unknown (⊤)
      // rather than falling through to the aggregate, which would silently return sibling fields'
      // shapes. See wala/ML#396.
      return null;
    }
    return this.getShapes(builder);
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<DType> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && asin.getConcreteType().getReference().equals(tuple)) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null && fieldIndex == index) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      asin.getConcreteType().getReference(),
                      findOrCreateAsciiAtom(fieldIndex.toString()),
                      Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                Set<DType> fieldDTypes =
                    this.getDTypesOfValue(builder, builder.getPointerAnalysis().getPointsToSet(pk));
                if (fieldDTypes != null) ret.addAll(fieldDTypes);
              }
            }
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
      // Per-index lookup saw a tuple but the requested field's PTS was empty — return UNKNOWN (⊤)
      // rather than falling through to the aggregate. See wala/ML#396.
      return EnumSet.of(DType.UNKNOWN);
    }
    return this.getDTypes(builder);
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> shapes = this.getShapesForIndex(builder, index);
    Set<DType> dTypes = this.getDTypesForIndex(builder, index);

    Set<TensorType> ret = HashSetFactory.make();

    // Null shapes signal "unknown per-index shape" (⊤). Emit one ⊤-shaped TensorType per dtype
    // rather than falling through to the aggregate {@code getTensorTypes}, which would silently
    // leak sibling fields' shapes. See wala/ML#396.
    if (shapes == null) {
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), null));
      return ret;
    }

    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));

    return ret;
  }

  /**
   * Shape-walk helper for the {@code tensors} argument of {@code
   * tf.data.Dataset.from_tensor_slices}.
   *
   * <p>{@code from_tensor_slices} accepts a "nested structure" argument: when the argument is a
   * tuple like {@code (a, b)}, the two elements are not the outer dimensions of a single tensor —
   * they are independent tensors bundled together, and the dataset yields tuples of per-element
   * slices. The base {@link TensorGenerator#getShapesOfValue} treats tuple and list identically
   * (both as multi-dim tensor values), which is the right interpretation for {@code
   * tf.convert_to_tensor((1, 2, 3))} and for nested-inside-list cases like {@code [(7, 8), (9,
   * 10)]}, but wrong for a top-level tuple argument to this specific API.
   *
   * <p>This helper special-cases a top-level tuple: walk each field independently and union the
   * per-field shapes (obtained via the base {@link TensorGenerator#getShapesOfValue}, which still
   * treats nested tuples as multi-dim). For non-tuple top-level values it falls through to the base
   * implementation unchanged, so list literals and single-tensor arguments keep their current
   * behavior. See wala/ML#366.
   */
  private Set<List<Dimension<?>>> getShapesOfTensorsArgument(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    boolean sawTuple = false;

    for (InstanceKey ik : valuePointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null || !asin.getConcreteType().getReference().equals(tuple)) {
        continue;
      }
      sawTuple = true;
      OrdinalSet<InstanceKey> objectCatalogPointsToSet =
          builder
              .getPointerAnalysis()
              .getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));

      LOGGER.fine(
          DatasetFromTensorSlicesGenerator.class.getName()
              + ".getShapesOfTensorsArgument: top-level tuple catalog size="
              + objectCatalogPointsToSet.size());

      for (InstanceKey catalogIK : objectCatalogPointsToSet) {
        Integer fieldIndex =
            getFieldIndex((com.ibm.wala.ipa.callgraph.propagation.ConstantKey<?>) catalogIK);
        if (fieldIndex == null) continue;
        FieldReference subscript =
            FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
        IField f = builder.getClassHierarchy().resolveField(subscript);
        if (f == null) continue;
        PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
        OrdinalSet<InstanceKey> fieldPts = builder.getPointerAnalysis().getPointsToSet(pk);
        Set<List<Dimension<?>>> fieldShapes = this.getShapesOfValue(builder, fieldPts);
        if (fieldShapes != null) ret.addAll(fieldShapes);
      }
    }

    if (!sawTuple) {
      return this.getShapesOfValue(builder, valuePointsToSet);
    }
    return ret;
  }

  /**
   * Dtype counterpart of {@link #getShapesOfTensorsArgument}. See that method for the rationale.
   */
  private Set<DType> getDTypesOfTensorsArgument(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    boolean sawTuple = false;

    for (InstanceKey ik : valuePointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null || !asin.getConcreteType().getReference().equals(tuple)) {
        continue;
      }
      sawTuple = true;
      OrdinalSet<InstanceKey> objectCatalogPointsToSet =
          builder
              .getPointerAnalysis()
              .getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));

      for (InstanceKey catalogIK : objectCatalogPointsToSet) {
        Integer fieldIndex =
            getFieldIndex((com.ibm.wala.ipa.callgraph.propagation.ConstantKey<?>) catalogIK);
        if (fieldIndex == null) continue;
        FieldReference subscript =
            FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
        IField f = builder.getClassHierarchy().resolveField(subscript);
        if (f == null) continue;
        PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
        OrdinalSet<InstanceKey> fieldPts = builder.getPointerAnalysis().getPointsToSet(pk);
        Set<DType> fieldDTypes = this.getDTypesOfValue(builder, fieldPts);
        if (fieldDTypes != null) ret.addAll(fieldDTypes);
      }
    }

    if (!sawTuple) {
      return this.getDTypesOfValue(builder, valuePointsToSet);
    }
    return ret;
  }
}
