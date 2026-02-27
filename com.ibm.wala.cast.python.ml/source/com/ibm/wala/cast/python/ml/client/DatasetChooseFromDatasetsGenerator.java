package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
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
 * A generator for tensors created by {@code tf.data.Dataset.choose_from_datasets}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetChooseFromDatasetsGenerator extends DatasetGenerator {

  /** Parameter indices for {@code tf.data.Dataset.choose_from_datasets}. */
  protected enum Parameters {
    /** A list of datasets to choose from. */
    DATASETS,
    /** A dataset of indices into {@code datasets}. */
    CHOICE_DATASET,
    /** Whether to stop when any dataset is empty. */
    STOP_ON_EMPTY_DATASET;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public DatasetChooseFromDatasetsGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetChooseFromDatasetsGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> datasetsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName());

    if (datasetsPTS != null && !datasetsPTS.isEmpty()) {
      Set<DType> ret = HashSetFactory.make();
      for (InstanceKey ik : datasetsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.getConcreteType().getReference().equals(list)
                || asin.getConcreteType().getReference().equals(tuple))) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            LOGGER.fine("    Processing list fieldIndex: " + fieldIndex);
            if (fieldIndex != null) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      asin.getConcreteType().getReference(),
                      findOrCreateAsciiAtom(fieldIndex.toString()),
                      Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                OrdinalSet<InstanceKey> fieldPTS = builder.getPointerAnalysis().getPointsToSet(pk);
                LOGGER.fine("      Field PTS size: " + fieldPTS.size());
                for (InstanceKey fieldIK : fieldPTS) {
                  LOGGER.fine("        Field element: " + fieldIK);
                }
                ret.addAll(this.getDTypesOfValue(builder, fieldPTS));
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
      return ret;
    }
    return Collections.emptySet();
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> datasetsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName());

    if (datasetsPTS != null && !datasetsPTS.isEmpty()) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (InstanceKey ik : datasetsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.getConcreteType().getReference().equals(list)
                || asin.getConcreteType().getReference().equals(tuple))) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            LOGGER.fine("    Processing list fieldIndex: " + fieldIndex);
            if (fieldIndex != null) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      asin.getConcreteType().getReference(),
                      findOrCreateAsciiAtom(fieldIndex.toString()),
                      Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                OrdinalSet<InstanceKey> fieldPTS = builder.getPointerAnalysis().getPointsToSet(pk);
                LOGGER.fine("      Field PTS size: " + fieldPTS.size());
                for (InstanceKey fieldIK : fieldPTS) {
                  LOGGER.fine("        Field element: " + fieldIK);
                }
                ret.addAll(this.getShapesOfValue(builder, fieldPTS));
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
      return ret;
    }
    return Collections.emptySet();
  }
}
