package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.withoutClientOnlyKeys;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by {@code tf.data.Dataset.sample_from_datasets}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetSampleFromDatasetsGenerator extends DatasetJoinGenerator {

  private static final Logger LOGGER =
      Logger.getLogger(DatasetSampleFromDatasetsGenerator.class.getName());

  /** Parameter indices for {@code tf.data.Dataset.sample_from_datasets}. */
  protected enum Parameters {
    /** A list of datasets to choose from. */
    DATASETS,
    /** A list of weights. */
    WEIGHTS,
    /** A random seed. */
    SEED,
    /** Whether to stop when any dataset is empty. */
    STOP_ON_EMPTY_DATASET;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  protected DatasetSampleFromDatasetsGenerator(PointsToSetVariable source) {
    super(source);
  }

  protected DatasetSampleFromDatasetsGenerator(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unless it stops at the first dataset that runs out, or a weight may leave a dataset
   *     unsampled, a sample yields every element of each dataset it is given: each size it may have
   *     is the sum over the list's positions of one size each, a position's sizes the union over
   *     the datasets it may hold, so a dataset given twice counts twice. The sizes are unknown when
   *     {@code stop_on_empty_dataset} may be anything but {@code False}, {@code weights} may be
   *     supplied, the list is not built in place by constant positions alone (as one appended to
   *     is), or a dataset's size is.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    int stopPosition = Parameters.STOP_ON_EMPTY_DATASET.getIndex();
    String stopName = Parameters.STOP_ON_EMPTY_DATASET.getName();
    Boolean supplied = this.isArgumentSyntacticallySupplied(builder, stopPosition, stopName);
    if (supplied == null) return Collections.emptySet();
    if (supplied) {
      Set<Boolean> stops =
          getPossibleBooleanValues(this.getArgumentPointsToSet(builder, stopPosition, stopName));
      if (stops == null || !stops.equals(Collections.singleton(Boolean.FALSE)))
        return Collections.emptySet();
    }
    // A weight of 0 leaves its dataset unsampled.
    if (!Boolean.FALSE.equals(
        this.isArgumentSyntacticallySupplied(
            builder, Parameters.WEIGHTS.getIndex(), Parameters.WEIGHTS.getName())))
      return Collections.emptySet();
    OrdinalSet<InstanceKey> sequences =
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName());
    if (sequences == null || sequences.isEmpty()) return Collections.emptySet();
    Set<Long> ret = HashSetFactory.make();
    for (InstanceKey sequence : sequences) {
      List<List<TensorGenerator>> positions =
          positionsOf(builder, sequence, TensorFlowTypes.DATASET_SAMPLE_FROM_DATASETS_TYPE);
      if (positions == null) return Collections.emptySet();
      Set<Long> sizes = null;
      for (List<TensorGenerator> position : positions) {
        Set<Long> at = sizesOfAny(builder, position);
        sizes = sizes == null ? at : sums(sizes, at);
        if (sizes.isEmpty()) return sizes;
      }
      if (sizes == null) return Collections.emptySet();
      ret.addAll(sizes);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The datasets the {@code datasets} argument holds, since the element is any one of
   *     their elements (wala/ML#1215), or {@code null} for a view fixed to one of them. The
   *     whole-element reads are this generator's own, over every dataset the list holds, so a view
   *     reads the union of the datasets there, not its own alone: sound, if imprecise.
   */
  @Override
  public List<TensorGenerator> getReceiverGenerators(PropagationCallGraphBuilder builder) {
    if (this.getFixedReceiver() != null) return null;
    return datasetsIn(
        builder,
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName()));
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> ret = HashSetFactory.make();

    // Bypass pointer analysis imprecision for list elements by directly analyzing the IR
    // if the list was created locally in the caller.
    PointerKey pk = this.getSource() != null ? this.getSource().getPointerKey() : null;
    if (pk instanceof LocalPointerKey) {
      CGNode caller = ((LocalPointerKey) pk).getNode();
      int vn = ((LocalPointerKey) pk).getValueNumber();
      SSAInstruction defInst = caller.getDU().getDef(vn);
      if (defInst instanceof PythonInvokeInstruction) {
        PythonInvokeInstruction call = (PythonInvokeInstruction) defInst;
        int paramPos = Parameters.DATASETS.getIndex();
        int argVn = -1;
        if (paramPos + 1 < call.getNumberOfUses()) {
          argVn = call.getUse(paramPos + 1);
        }
        if (argVn != -1) {
          DefUse du = caller.getDU();
          SSAInstruction def = du.getDef(argVn);
          if (def instanceof SSANewInstruction) {
            Iterator<SSAInstruction> uses = du.getUses(argVn);
            while (uses.hasNext()) {
              SSAInstruction use = uses.next();
              if (use instanceof AstPropertyWrite) {
                AstPropertyWrite write = (AstPropertyWrite) use;
                if (write.getObjectRef() == argVn) {
                  int valueVn = write.getValue();
                  PointerKey valuePK =
                      builder
                          .getPointerAnalysis()
                          .getHeapModel()
                          .getPointerKeyForLocal(caller, valueVn);
                  if (!builder.getPropagationSystem().isImplicit(valuePK)) {
                    PointsToSetVariable var =
                        builder.getPropagationSystem().findOrCreatePointsToSet(valuePK);
                    try {
                      TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
                      if (generator != null) {
                        Set<DType> generatorDTypes = memoizedDTypes(builder, generator);
                        if (generatorDTypes != null) ret.addAll(generatorDTypes);
                      }
                    } catch (IllegalArgumentException e) {
                      // Ignore.
                    }
                  }
                }
              }
            }
            if (!ret.isEmpty()) return ret;
          }
        }
      }
    }

    OrdinalSet<InstanceKey> datasetsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName());

    if (datasetsPTS != null && !datasetsPTS.isEmpty()) {
      for (InstanceKey ik : datasetsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.concreteType().getReference().equals(list)
                || asin.concreteType().getReference().equals(tuple))) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              withoutClientOnlyKeys(
                  builder
                      .getPointerAnalysis()
                      .getPointsToSet(
                          ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                              .getPointerKeyForObjectCatalog(asin)));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            LOGGER.fine("    Processing list fieldIndex: " + fieldIndex);
            if (fieldIndex != null) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      asin.concreteType().getReference(),
                      findOrCreateAsciiAtom(fieldIndex.toString()),
                      Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey fieldPK = builder.getPointerKeyForInstanceField(asin, f);

                boolean preciseTypesFound = false;
                if (!builder.getPropagationSystem().isImplicit(fieldPK)) {
                  PointsToSetVariable var =
                      builder.getPropagationSystem().findOrCreatePointsToSet(fieldPK);
                  try {
                    TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
                    if (generator != null) {
                      Set<DType> preciseTypes = memoizedDTypes(builder, generator);
                      if (preciseTypes != null && !preciseTypes.isEmpty()) {
                        ret.addAll(preciseTypes);
                        preciseTypesFound = true;
                      }
                    }
                  } catch (IllegalArgumentException e) {
                    // Ignore
                  }
                }

                if (!preciseTypesFound) {
                  OrdinalSet<InstanceKey> fieldPTS =
                      withoutClientOnlyKeys(builder.getPointerAnalysis().getPointsToSet(fieldPK));
                  ret.addAll(this.getDTypesOfValue(builder, fieldPTS));
                }
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
    return null;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    // Bypass pointer analysis imprecision for list elements by directly analyzing the IR
    // if the list was created locally in the caller.
    PointerKey pk = this.getSource() != null ? this.getSource().getPointerKey() : null;
    if (pk instanceof LocalPointerKey) {
      CGNode caller = ((LocalPointerKey) pk).getNode();
      int vn = ((LocalPointerKey) pk).getValueNumber();
      SSAInstruction defInst = caller.getDU().getDef(vn);
      if (defInst instanceof PythonInvokeInstruction) {
        PythonInvokeInstruction call = (PythonInvokeInstruction) defInst;
        int paramPos = Parameters.DATASETS.getIndex();
        int argVn = -1;
        if (paramPos + 1 < call.getNumberOfUses()) {
          argVn = call.getUse(paramPos + 1);
        }
        if (argVn != -1) {
          DefUse du = caller.getDU();
          SSAInstruction def = du.getDef(argVn);
          if (def instanceof SSANewInstruction) {
            Iterator<SSAInstruction> uses = du.getUses(argVn);
            while (uses.hasNext()) {
              SSAInstruction use = uses.next();
              if (use instanceof AstPropertyWrite) {
                AstPropertyWrite write = (AstPropertyWrite) use;
                if (write.getObjectRef() == argVn) {
                  int valueVn = write.getValue();
                  PointerKey valuePK =
                      builder
                          .getPointerAnalysis()
                          .getHeapModel()
                          .getPointerKeyForLocal(caller, valueVn);
                  if (!builder.getPropagationSystem().isImplicit(valuePK)) {
                    PointsToSetVariable var =
                        builder.getPropagationSystem().findOrCreatePointsToSet(valuePK);
                    try {
                      TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
                      if (generator != null) {
                        ret.addAll(memoizedShapeResult(builder, generator).members());
                      }
                    } catch (IllegalArgumentException e) {
                      // Ignore.
                    }
                  }
                }
              }
            }
            if (!ret.isEmpty()) return ret;
          }
        }
      }
    }

    OrdinalSet<InstanceKey> datasetsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.DATASETS.getIndex(), Parameters.DATASETS.getName());

    if (datasetsPTS != null && !datasetsPTS.isEmpty()) {
      for (InstanceKey ik : datasetsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.concreteType().getReference().equals(list)
                || asin.concreteType().getReference().equals(tuple))) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              withoutClientOnlyKeys(
                  builder
                      .getPointerAnalysis()
                      .getPointsToSet(
                          ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                              .getPointerKeyForObjectCatalog(asin)));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            LOGGER.fine("    Processing list fieldIndex: " + fieldIndex);
            if (fieldIndex != null) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      asin.concreteType().getReference(),
                      findOrCreateAsciiAtom(fieldIndex.toString()),
                      Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey fieldPK = builder.getPointerKeyForInstanceField(asin, f);

                boolean preciseTypesFound = false;
                if (!builder.getPropagationSystem().isImplicit(fieldPK)) {
                  PointsToSetVariable var =
                      builder.getPropagationSystem().findOrCreatePointsToSet(fieldPK);
                  try {
                    TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
                    if (generator != null) {
                      Set<List<Dimension<?>>> preciseTypes =
                          memoizedShapeResult(builder, generator).members();
                      if (!preciseTypes.isEmpty()) {
                        ret.addAll(preciseTypes);
                        preciseTypesFound = true;
                      }
                    }
                  } catch (IllegalArgumentException e) {
                    // Ignore
                  }
                }

                if (!preciseTypesFound) {
                  OrdinalSet<InstanceKey> fieldPTS =
                      withoutClientOnlyKeys(builder.getPointerAnalysis().getPointsToSet(fieldPK));
                  if (!addShapesOrTop(ret, this.getShapesOfValue(builder, fieldPTS))) return null;
                }
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
      return ret.isEmpty() ? null : ret;
    }
    return null;
  }
}
