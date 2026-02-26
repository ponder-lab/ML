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
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by {@code tf.data.Dataset.from_tensors}.
 *
 * <p>Unlike {@code from_tensor_slices}, which slices the input along its first dimension, {@code
 * from_tensors} creates a dataset containing the input as a single, whole element.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetFromTensorsGenerator extends DatasetGenerator implements TupleElementProvider {

  /** Parameter indices for {@code tf.data.Dataset.from_tensors}. */
  protected enum Parameters {
    /** The tensor or structured object to be converted into a dataset. */
    TENSORS,
    /** The name of the operation (optional). */
    NAME;

    public String getName() {
      return name().toLowerCase();
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
  public DatasetFromTensorsGenerator(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a new {@code DatasetFromTensorsGenerator}.
   *
   * @param node the call graph node where the dataset is created
   */
  public DatasetFromTensorsGenerator(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation handles structured elements (tuples or lists) by returning the
   *     union of the types of their constituent members, instead of treating the structure itself
   *     as a single tensor type.
   */
  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<TensorType> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.getConcreteType().getReference().equals(tuple)
                || asin.getConcreteType().getReference().equals(list))) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

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
          Set<DType> dTypes = this.getDTypesOfValue(builder, singletonPTS);
          for (List<Dimension<?>> shape : shapes) {
            for (DType dtype : dTypes) {
              ret.add(new TensorType(dtype.name().toLowerCase(), shape));
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
        if (asin != null
            && (asin.getConcreteType().getReference().equals(tuple)
                || asin.getConcreteType().getReference().equals(list))) {
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
                      Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                ret.addAll(
                    this.getShapesOfValue(
                        builder, builder.getPointerAnalysis().getPointsToSet(pk)));
              }
            }
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
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
        if (asin != null
            && (asin.getConcreteType().getReference().equals(tuple)
                || asin.getConcreteType().getReference().equals(list))) {
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
                      Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                ret.addAll(
                    this.getDTypesOfValue(
                        builder, builder.getPointerAnalysis().getPointsToSet(pk)));
              }
            }
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }
    return this.getDTypes(builder);
  }

  /** {@inheritDoc} */
  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> shapes = this.getShapesForIndex(builder, index);
    Set<DType> dTypes = this.getDTypesForIndex(builder, index);

    Set<TensorType> ret = HashSetFactory.make();

    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));

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
        if (asin != null
            && (asin.getConcreteType().getReference().equals(tuple)
                || asin.getConcreteType().getReference().equals(list))) {
          // It's a structured element. We return the union of all possible shapes of its members.
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                ret.addAll(
                    this.getShapesOfValue(
                        builder, builder.getPointerAnalysis().getPointsToSet(pk)));
              }
            }
          }
        } else {
          ret.addAll(
              this.getShapesOfValue(
                  builder,
                  OrdinalSet.toOrdinalSet(
                      Collections.singleton(ik),
                      builder.getPointerAnalysis().getInstanceKeyMapping())));
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }
    return Collections.emptySet();
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

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<DType> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.getConcreteType().getReference().equals(tuple)
                || asin.getConcreteType().getReference().equals(list))) {
          // It's a structured element. We return the union of all possible dtypes of its members.
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                ret.addAll(
                    this.getDTypesOfValue(
                        builder, builder.getPointerAnalysis().getPointsToSet(pk)));
              }
            }
          }
        } else {
          ret.addAll(
              this.getDTypesOfValue(
                  builder,
                  OrdinalSet.toOrdinalSet(
                      Collections.singleton(ik),
                      builder.getPointerAnalysis().getInstanceKeyMapping())));
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
    }
    return Collections.emptySet();
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
}
