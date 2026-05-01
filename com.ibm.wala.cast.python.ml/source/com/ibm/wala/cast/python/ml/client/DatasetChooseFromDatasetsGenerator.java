package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
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
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by {@code tf.data.Dataset.choose_from_datasets}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetChooseFromDatasetsGenerator extends DatasetGenerator {

  private static final Logger LOGGER =
      Logger.getLogger(DatasetChooseFromDatasetsGenerator.class.getName());

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
                        ret.addAll(generator.getDTypes(builder));
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
                PointerKey fieldPK = builder.getPointerKeyForInstanceField(asin, f);

                boolean preciseTypesFound = false;
                if (!builder.getPropagationSystem().isImplicit(fieldPK)) {
                  PointsToSetVariable var =
                      builder.getPropagationSystem().findOrCreatePointsToSet(fieldPK);
                  try {
                    TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
                    if (generator != null) {
                      Set<DType> preciseTypes = generator.getDTypes(builder);
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
                      builder.getPointerAnalysis().getPointsToSet(fieldPK);
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
    return EnumSet.of(DType.UNKNOWN);
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
                        Set<List<Dimension<?>>> generatorShapes = generator.getShapes(builder);
                        if (generatorShapes != null) ret.addAll(generatorShapes);
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
                PointerKey fieldPK = builder.getPointerKeyForInstanceField(asin, f);

                boolean preciseTypesFound = false;
                if (!builder.getPropagationSystem().isImplicit(fieldPK)) {
                  PointsToSetVariable var =
                      builder.getPropagationSystem().findOrCreatePointsToSet(fieldPK);
                  try {
                    TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
                    if (generator != null) {
                      Set<List<Dimension<?>>> preciseTypes = generator.getShapes(builder);
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
                      builder.getPointerAnalysis().getPointsToSet(fieldPK);
                  ret.addAll(this.getShapesOfValue(builder, fieldPTS));
                }
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
      return ret.isEmpty() ? null : ret;
    }
    return null;
  }
}
