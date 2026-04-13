package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by {@code tf.data.Dataset.batch}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetBatchGenerator extends DatasetGenerator {

  protected enum Parameters {
    BATCH_SIZE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public DatasetBatchGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetBatchGenerator(CGNode node) {
    super(node);
  }

  private OrdinalSet<InstanceKey> getReceiverPTS(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (pts != null && !pts.isEmpty()) {
      return pts;
    }
    return getReceiverFromCallSite(builder);
  }

  private OrdinalSet<InstanceKey> getReceiverFromCallSite(PropagationCallGraphBuilder builder) {
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int funcVn = call.getUse(0);
      CGNode caller = this.getNode();
      DefUse du = caller.getDU();
      SSAInstruction def = du.getDef(funcVn);
      if (def instanceof SSAGetInstruction) {
        int objVn = ((SSAGetInstruction) def).getRef();
        PointerKey objKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, objVn);
        return builder.getPointerAnalysis().getPointsToSet(objKey);
      } else if (def instanceof PythonPropertyRead) {
        int objVn = ((PythonPropertyRead) def).getObjectRef();
        PointerKey objKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, objVn);
        return builder.getPointerAnalysis().getPointsToSet(objKey);
      }
    }
    return OrdinalSet.empty();
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Get the shapes from the input dataset.
    OrdinalSet<InstanceKey> receiverPTS = getReceiverPTS(builder);

    Set<List<Dimension<?>>> inputShapes = null;
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      inputShapes = this.getShapesOfValue(builder, receiverPTS);
    }

    // Lattice propagation: ⊤ in → ⊤ out; ⊥ in → ⊥ out.
    if (inputShapes == null) return null;
    if (inputShapes.isEmpty()) return Collections.emptySet();

    // Get the batch size.
    Set<Long> batchSizes =
        getPossibleLongValues(
            this.getArgumentPointsToSet(
                builder, Parameters.BATCH_SIZE.getIndex(), Parameters.BATCH_SIZE.getName()));

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    // Get the total dataset sizes (before batching).
    Set<Long> datasetSizes = this.getDatasetSizes(builder);

    for (List<Dimension<?>> shape : inputShapes) {
      if (batchSizes.isEmpty()) {
        // If the batch size is unknown, we assume it is symbolic.
        List<Dimension<?>> newShape = new ArrayList<>();
        newShape.add(new SymbolicDim("?"));
        newShape.addAll(shape);
        ret.add(newShape);
      } else {
        for (Long batchSize : batchSizes) {
          if (batchSize != null) {
            // Drop remainder is implicitly false here, so we could have a partial batch.
            // If we know the dataset size, we can infer if a partial batch is possible.
            boolean canHaveFullBatch;
            // Accumulate every possible partial-batch size across the dataset sizes instead of
            // overwriting (previously only the last non-divisor remainder was retained).
            Set<Long> partialBatchSizes = new HashSet<>();
            boolean partialBatchSizeUnknown = false;

            if (!datasetSizes.isEmpty()) {
              canHaveFullBatch = false;
              for (Long dsSize : datasetSizes) {
                if (dsSize >= batchSize) {
                  canHaveFullBatch = true;
                }
                long rem = dsSize % batchSize;
                if (rem != 0) {
                  partialBatchSizes.add(rem);
                }
              }
            } else {
              // We don't know the dataset size; assume both a full batch and a symbolic partial
              // batch are possible.
              canHaveFullBatch = true;
              partialBatchSizeUnknown = true;
            }

            if (canHaveFullBatch) {
              List<Dimension<?>> newShape = new ArrayList<>();
              newShape.add(new NumericDim(batchSize.intValue()));
              newShape.addAll(shape);
              ret.add(newShape);
            }

            for (Long partialBatchSize : partialBatchSizes) {
              List<Dimension<?>> newShape = new ArrayList<>();
              newShape.add(new NumericDim(partialBatchSize.intValue()));
              newShape.addAll(shape);
              ret.add(newShape);
            }

            if (partialBatchSizeUnknown) {
              List<Dimension<?>> newShape = new ArrayList<>();
              newShape.add(new SymbolicDim("?"));
              newShape.addAll(shape);
              ret.add(newShape);
            }

          } else {
            List<Dimension<?>> newShape = new ArrayList<>();
            newShape.add(new SymbolicDim("?"));
            newShape.addAll(shape);
            ret.add(newShape);
          }
        }
      }
    }

    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> receiverPTS = getReceiverPTS(builder);
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      return this.getDTypesOfValue(builder, receiverPTS);
    }
    return EnumSet.of(DType.UNKNOWN);
  }
}
