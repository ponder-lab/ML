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
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by <code>tf.data.Dataset.batch</code>.
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

  private OrdinalSet<InstanceKey> getReceiverPTS(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, "self");
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

    Set<List<Dimension<?>>> inputShapes = Collections.emptySet();
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      inputShapes = this.getShapesOfValue(builder, receiverPTS);
    }

    if (inputShapes.isEmpty()) {
      return Collections.emptySet();
    }

    // Get the batch size.
    Set<Long> batchSizes =
        getPossibleLongValues(
            this.getArgumentPointsToSet(
                builder, Parameters.BATCH_SIZE.getIndex(), Parameters.BATCH_SIZE.getName()));

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    for (List<Dimension<?>> shape : inputShapes) {
      if (batchSizes.isEmpty()) {
        // If the batch size is unknown, we assume it is symbolic.
        List<Dimension<?>> newShape = new ArrayList<>();
        newShape.add(new SymbolicDim("?"));
        newShape.addAll(shape);
        ret.add(newShape);
      } else {
        for (Long batchSize : batchSizes) {
          List<Dimension<?>> newShape = new ArrayList<>();
          if (batchSize != null) {
            newShape.add(new NumericDim(batchSize.intValue()));
          } else {
            newShape.add(new SymbolicDim("?"));
          }
          newShape.addAll(shape);
          ret.add(newShape);
        }
      }
    }

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> receiverPTS = getReceiverPTS(builder);
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      return this.getDTypesOfValue(builder, receiverPTS);
    }
    return Collections.emptySet();
  }
}
