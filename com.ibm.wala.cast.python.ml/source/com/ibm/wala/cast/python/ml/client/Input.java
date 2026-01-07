package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;
import static java.util.Collections.emptySet;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by the `Input()` function in TensorFlow/Keras.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Input">TensorFlow Input()
 *     API</a>.
 */
public class Input extends TensorGenerator {

  private static final Logger LOGGER = getLogger(Input.class.getName());

  private static final int SHAPE_PARAMETER_POSITION = 0;
  private static final int BATCH_SIZE_PARAMETER_POSITION = 1;
  private static final int DTYPE_PARAMETER_POSITION = 3;

  public Input(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    LOGGER.info(
        "No dtype specified for source: " + source + ". Using default dtype of: " + FLOAT32 + " .");

    // Use the default dtype of float32.
    return EnumSet.of(FLOAT32);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return emptySet();
  }

  @Override
  protected EnumSet<DType> getDTypes(PropagationCallGraphBuilder builder) {
    int valNum = getArgumentValueNumber(builder, DTYPE_PARAMETER_POSITION, true);
    OrdinalSet<InstanceKey> pointsToSet = null;
    if (valNum > 0) {
      PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), valNum);
      pointsToSet = pa.getPointsToSet(pk);
    }

    if (pointsToSet == null || pointsToSet.isEmpty()) {
      pointsToSet = getKeywordArgumentPointsToSet(builder, "dtype");
    }

    if (pointsToSet != null && !pointsToSet.isEmpty()) {
      return getDTypesFromDTypeArgument(builder, pointsToSet);
    }
    return getDefaultDTypes(builder);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    int shapeValNum = getArgumentValueNumber(builder, SHAPE_PARAMETER_POSITION, true);
    OrdinalSet<InstanceKey> shapePts = null;
    if (shapeValNum > 0) {
      PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), shapeValNum);
      shapePts = pa.getPointsToSet(pk);
    }

    if (shapePts == null || shapePts.isEmpty()) {
      shapePts = getKeywordArgumentPointsToSet(builder, "shape");
    }

    Set<List<Dimension<?>>> shapes;
    if (shapePts != null && !shapePts.isEmpty()) {
      shapes = getShapesFromShapeArgument(builder, shapePts);
    } else {
      shapes = getDefaultShapes(builder);
    }

    // Handle batch_size
    int batchSizeValNum = getArgumentValueNumber(builder, BATCH_SIZE_PARAMETER_POSITION, true);

    Set<Long> batchSizes = new HashSet<>();
    if (batchSizeValNum > 0) {
      PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
      PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(this.getNode(), batchSizeValNum);
      OrdinalSet<InstanceKey> pts = pa.getPointsToSet(pk);
      batchSizes.addAll(getPossibleLongArguments(pts));
    }

    // Also check for batch_size keyword
    OrdinalSet<InstanceKey> batchSizePts = getKeywordArgumentPointsToSet(builder, "batch_size");
    if (batchSizePts != null && !batchSizePts.isEmpty()) {
      batchSizes.addAll(getPossibleLongArguments(batchSizePts));
    }

    if (batchSizes.isEmpty()) {
      batchSizes.add(null);
    }

    Set<List<Dimension<?>>> newShapes = HashSetFactory.make();
    for (List<Dimension<?>> shape : shapes) {
      for (Long batchSize : batchSizes) {
        List<Dimension<?>> newShape = new ArrayList<>();
        // Prepend batch size.
        if (batchSize != null) {
          newShape.add(new NumericDim(batchSize.intValue()));
        } else {
          newShape.add(null);
        }
        newShape.addAll(shape);
        newShapes.add(newShape);
      }
    }
    return newShapes;
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE_PARAMETER_POSITION;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_PARAMETER_POSITION;
  }

  private OrdinalSet<InstanceKey> getKeywordArgumentPointsToSet(
      PropagationCallGraphBuilder builder, String keyword) {
    OrdinalSet<InstanceKey> result = null;
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();

    CallString cs = (CallString) this.getNode().getContext().get(CALL_STRING);

    if (cs == null || cs.getCallSiteRefs().length == 0) {
      return null;
    }

    CallSiteReference siteReference = cs.getCallSiteRefs()[0];

    for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(this.getNode());
        it.hasNext(); ) {
      CGNode caller = it.next();
      SSAAbstractInvokeInstruction[] calls = caller.getIR().getCalls(siteReference);

      for (SSAAbstractInvokeInstruction call : calls) {
        if (call instanceof PythonInvokeInstruction) {
          PythonInvokeInstruction pyCall = (PythonInvokeInstruction) call;
          int use = pyCall.getUse(keyword);
          if (use != -1) {
            PointerKey pk = pa.getHeapModel().getPointerKeyForLocal(caller, use);
            OrdinalSet<InstanceKey> pts = pa.getPointsToSet(pk);
            if (result == null) {
              result = pts;
            } else {
              result = OrdinalSet.unify(result, pts);
            }
          }
        }
      }
    }
    return result;
  }

  private Set<Long> getPossibleLongArguments(OrdinalSet<InstanceKey> pointsToSet) {
    Set<Long> ret = HashSetFactory.make();
    if (pointsToSet == null) return ret;

    for (InstanceKey instanceKey : pointsToSet) {

      if (instanceKey instanceof ConstantKey) {
        ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
        Object constantKeyValue = constantKey.getValue();
        if (constantKeyValue instanceof Long) {
          ret.add((Long) constantKeyValue);
        } else if (constantKeyValue instanceof Integer) {
          ret.add(((Integer) constantKeyValue).longValue());
        } else if (constantKeyValue == null) {
          ret.add(null);
        }
      }
    }
    return ret;
  }
}
