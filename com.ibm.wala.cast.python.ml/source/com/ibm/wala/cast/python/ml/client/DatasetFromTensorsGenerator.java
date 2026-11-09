package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.withoutClientOnlyKeys;

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
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by {@code tf.data.Dataset.from_tensors}.
 *
 * <p>Unlike {@code from_tensor_slices}, which slices the input along its first dimension, {@code
 * from_tensors} creates a dataset containing the input as a single, whole element.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetFromTensorsGenerator extends DatasetGenerator implements TupleElementProvider {

  private static final Logger LOGGER =
      Logger.getLogger(DatasetFromTensorsGenerator.class.getName());

  /** Parameter indices for {@code tf.data.Dataset.from_tensors}. */
  protected enum Parameters {
    /** The tensor or structured object to be converted into a dataset. */
    TENSORS,
    /** The name of the operation (optional). */
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  /**
   * Constructs a new {@code DatasetFromTensorsGenerator}.
   *
   * @param source the points-to set variable representing the source of the dataset
   */
  protected DatasetFromTensorsGenerator(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a new {@code DatasetFromTensorsGenerator}.
   *
   * @param node the call graph node where the dataset is created
   */
  protected DatasetFromTensorsGenerator(CGNode node) {
    super(node);
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && isPositional(asin.concreteType().getReference())) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The length of the literal tuple the {@code tensors} argument holds at the path, which
   *     the dataset's one element holds whole.
   */
  @Override
  public OptionalInt declaredArity(PropagationCallGraphBuilder builder, List<Object> path) {
    return TupleElementProvider.declaredArityOf(builder, this.componentPointsToSet(builder, path));
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A path reaches a component of the {@code tensors} argument, which the dataset's one
   *     element holds whole (wala/ML#1211). The dataset has no receiver to read it from.
   */
  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer) return this.yieldsTuple(builder);
    OrdinalSet<InstanceKey> component = this.componentPointsToSet(builder, path);
    return component != null && !component.isEmpty();
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The component is a structure when the {@code tensors} argument holds only tuples,
   *     dicts and namedtuples there.
   */
  @Override
  public boolean isStructureAt(PropagationCallGraphBuilder builder, List<Object> path) {
    return TupleElementProvider.holdsOnlyStructures(
        builder, this.componentPointsToSet(builder, path));
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The shapes of the component of the {@code tensors} argument at the path, unsliced,
   *     since the dataset's one element is the argument whole (wala/ML#1211).
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getShapesForIndex(builder, index);
    OrdinalSet<InstanceKey> component = this.componentPointsToSet(builder, path);
    if (component == null || component.isEmpty()) return null;
    Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, component);
    return shapes == null || shapes.isEmpty() ? null : shapes;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The dtypes of the component of the {@code tensors} argument at the path
   *     (wala/ML#1211).
   */
  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getDTypesForIndex(builder, index);
    OrdinalSet<InstanceKey> component = this.componentPointsToSet(builder, path);
    if (component == null || component.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    Set<DType> dtypes = this.getDTypesOfValue(builder, component);
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * The points-to set of the component a path of selectors reaches within the {@code tensors}
   * argument: an {@link Integer} selects a tuple's field by index, a {@link String} a dict's field
   * by key.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for PA lookup.
   * @param path The selectors, outermost first.
   * @return The component's points-to set, or {@code null} when a step reaches nothing.
   */
  private OrdinalSet<InstanceKey> componentPointsToSet(
      PropagationCallGraphBuilder builder, List<Object> path) {
    return selectComponent(
        builder,
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName()),
        path);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation handles structured elements (tuples or lists) by returning the
   *     union of the types of their constituent members, instead of treating the structure itself
   *     as a single tensor type.
   */
  @Override
  protected Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<TensorType> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && isPositional(asin.concreteType().getReference())) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              withoutClientOnlyKeys(
                  builder
                      .getPointerAnalysis()
                      .getPointsToSet(
                          ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                              .getPointerKeyForObjectCatalog(asin)));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null) {
              ret.addAll(this.getTensorTypesForIndex(builder, fieldIndex));
            }
          }
        } else {
          OrdinalSet<InstanceKey> singletonPTS =
              OrdinalSet.toOrdinalSet(
                  Collections.singleton(ik), builder.getPointerAnalysis().getInstanceKeyMapping());
          Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, singletonPTS);
          // An unknown shape (null, wala/ML#978) is one type of unknown rank, not a dereference.
          if (shapes == null) shapes = Collections.singleton(null);
          Set<DType> dTypes = dTypesOrTop(this.getDTypesOfValue(builder, singletonPTS));
          for (List<Dimension<?>> shape : shapes) {
            for (DType dtype : dTypes) {
              ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), shape));
            }
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }
    return super.getTensorTypes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation retrieves the shape of the constituent element at the specified
   *     index when the input to {@code from_tensors} is a structured object (tuple or list).
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && isPositional(asin.concreteType().getReference())) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              withoutClientOnlyKeys(
                  builder
                      .getPointerAnalysis()
                      .getPointsToSet(
                          ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                              .getPointerKeyForObjectCatalog(asin)));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null && fieldIndex == index) {
              FieldReference subscript =
                  positionField(asin.concreteType().getReference(), fieldIndex);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                if (!addShapesOrTop(
                    ret,
                    this.getShapesOfValue(
                        builder,
                        withoutClientOnlyKeys(builder.getPointerAnalysis().getPointsToSet(pk)))))
                  return null;
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

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation retrieves the dtype of the constituent element at the specified
   *     index when the input to {@code from_tensors} is a structured object (tuple or list).
   */
  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<DType> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && isPositional(asin.concreteType().getReference())) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              withoutClientOnlyKeys(
                  builder
                      .getPointerAnalysis()
                      .getPointsToSet(
                          ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                              .getPointerKeyForObjectCatalog(asin)));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null && fieldIndex == index) {
              FieldReference subscript =
                  positionField(asin.concreteType().getReference(), fieldIndex);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                ret.addAll(
                    dTypesOrTop(
                        this.getDTypesOfValue(
                            builder,
                            withoutClientOnlyKeys(
                                builder.getPointerAnalysis().getPointsToSet(pk)))));
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

  /** {@inheritDoc} */
  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> shapes = this.getShapesForIndex(builder, index);
    Set<DType> dTypes = this.getDTypesForIndex(builder, index);

    Set<TensorType> ret = HashSetFactory.make();

    // Null shapes signal "unknown per-index shape" (⊤). Emit one ⊤-shaped TensorType per dtype
    // rather than falling through to the aggregate {@code getTensorTypes}, which would silently
    // leak sibling fields' shapes. See wala/ML#396.
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

  /**
   * {@inheritDoc}
   *
   * @implNote For {@code tf.data.Dataset.from_tensors(tensors)}, the dataset contains a single
   *     element which is the {@code tensors} argument itself. If that argument is a structured
   *     object (tuple or list), this method returns the union of the shapes of its members.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // For tf.data.Dataset.from_tensors(tensors), the dataset contains a single element
    // which is the 'tensors' argument itself.
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && isPositional(asin.concreteType().getReference())) {
          // It's a structured element. We return the union of all possible shapes of its members.
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              withoutClientOnlyKeys(
                  builder
                      .getPointerAnalysis()
                      .getPointsToSet(
                          ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                              .getPointerKeyForObjectCatalog(asin)));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null) {
              FieldReference subscript =
                  positionField(asin.concreteType().getReference(), fieldIndex);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                if (!addShapesOrTop(
                    ret,
                    this.getShapesOfValue(
                        builder,
                        withoutClientOnlyKeys(builder.getPointerAnalysis().getPointsToSet(pk)))))
                  return null;
              }
            }
          }
        } else {
          if (!addShapesOrTop(
              ret,
              this.getShapesOfValue(
                  builder,
                  OrdinalSet.toOrdinalSet(
                      Collections.singleton(ik),
                      builder.getPointerAnalysis().getInstanceKeyMapping())))) return null;
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }
    return null;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote For {@code tf.data.Dataset.from_tensors(tensors)}, the dataset contains a single
   *     element which is the {@code tensors} argument itself. If that argument is a structured
   *     object (tuple or list), this method returns the union of the dtypes of its members.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    LOGGER.fine(
        DatasetFromTensorsGenerator.class.getName()
            + ".getDefaultDTypes: tensorsPTS size="
            + (tensorsPTS != null ? tensorsPTS.size() : "null"));

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<DType> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        LOGGER.fine(
            DatasetFromTensorsGenerator.class.getName() + ": processing ik=" + describe(ik));
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && isPositional(asin.concreteType().getReference())) {
          // It's a structured element. We return the union of all possible dtypes of its members.
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              withoutClientOnlyKeys(
                  builder
                      .getPointerAnalysis()
                      .getPointsToSet(
                          ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                              .getPointerKeyForObjectCatalog(asin)));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null) {
              FieldReference subscript =
                  positionField(asin.concreteType().getReference(), fieldIndex);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                ret.addAll(
                    dTypesOrTop(
                        this.getDTypesOfValue(
                            builder,
                            withoutClientOnlyKeys(
                                builder.getPointerAnalysis().getPointsToSet(pk)))));
              }
            }
          }
        } else {
          ret.addAll(
              dTypesOrTop(
                  this.getDTypesOfValue(
                      builder,
                      OrdinalSet.toOrdinalSet(
                          Collections.singleton(ik),
                          builder.getPointerAnalysis().getInstanceKeyMapping()))));
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }
    return EnumSet.of(DType.UNKNOWN);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote For {@code from_tensors}, the dataset always contains exactly one element.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(1L);
  }

  /**
   * A value's dtypes, or unknown (⊤) where they do not resolve ({@code null}), as {@link
   * #getDTypesForPath} reads a component's, so a value whose dtypes do not resolve is a tensor of
   * unknown dtype rather than a dereference.
   *
   * @param dtypes The value's dtypes, or {@code null}.
   * @return The dtypes, or {@link DType#UNKNOWN} alone.
   */
  private static Set<DType> dTypesOrTop(Set<DType> dtypes) {
    return dtypes == null ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }
}
