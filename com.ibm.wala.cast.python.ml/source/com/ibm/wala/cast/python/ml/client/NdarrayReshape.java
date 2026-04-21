package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Modeling of the NumPy {@code ndarray.reshape} method. Unlike {@link Reshape} (which models the
 * {@code tf.reshape} function and takes an explicit tensor argument), {@code ndarray.reshape} is a
 * method call &mdash; the tensor is the receiver, captured in the function's closure. The shape
 * argument is the first positional argument to the call.
 *
 * <p>Resolves {@code -1} in the target shape by dividing the receiver's total element count by the
 * product of known dims, matching {@link Reshape}'s algorithm. Dispatched via class-type check in
 * {@link TensorGeneratorFactory#getGenerator} keyed on {@code
 * NumpyTypes.RESHAPE_METHOD.getDeclaringClass()}, which keeps the dispatch disjoint from {@code
 * tf.reshape} despite the shared method name.
 */
public class NdarrayReshape extends TensorGenerator {

  @SuppressWarnings("unused")
  private static final Logger LOGGER = Logger.getLogger(NdarrayReshape.class.getName());

  /** Position of the shape argument in the invoke (first positional argument). */
  private static final int SHAPE_ARG_POSITION = 0;

  private static final String SHAPE_ARG_NAME = "shape";

  public NdarrayReshape(PointsToSetVariable source) {
    super(source);
  }

  private int getReceiverVn() {
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int funcVn = call.getUse(0);
      SSAInstruction funcDef = getNode().getDU().getDef(funcVn);
      if (funcDef instanceof PythonPropertyRead) {
        return ((PythonPropertyRead) funcDef).getObjectRef();
      }
    }
    return getArgumentValueNumber(RECEIVER_PARAMETER_POSITION);
  }

  @Override
  public Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    // Resolve the target shape from the `shape` argument, applying the `-1` inference rule by
    // dividing the receiver's known total size by the product of the explicit dims.
    OrdinalSet<InstanceKey> shapePts =
        this.getArgumentPointsToSet(builder, SHAPE_ARG_POSITION, SHAPE_ARG_NAME);
    if (shapePts == null || shapePts.isEmpty()) return getDefaultShapes(builder);

    Set<List<Dimension<?>>> rawShapes = this.getShapesFromShapeArgument(builder, shapePts);
    if (rawShapes.isEmpty()) return getDefaultShapes(builder);

    Set<List<Dimension<?>>> refinedShapes = HashSetFactory.make();

    for (List<Dimension<?>> shape : rawShapes) {
      int unknownIndex = -1;
      long productKnown = 1;
      boolean canInfer = true;

      for (int i = 0; i < shape.size(); i++) {
        Dimension<?> dim = shape.get(i);
        if (dim instanceof NumericDim) {
          int val = ((NumericDim) dim).value();
          if (val == -1) {
            if (unknownIndex != -1) {
              canInfer = false;
              break;
            }
            unknownIndex = i;
          } else {
            productKnown *= val;
          }
        } else {
          canInfer = false;
          break;
        }
      }

      if (unknownIndex != -1) {
        Set<List<Dimension<?>>> inputShapes = this.getDefaultShapes(builder);

        if (canInfer && inputShapes != null && !inputShapes.isEmpty()) {
          for (List<Dimension<?>> inputShape : inputShapes) {
            long inputSize = 1;
            boolean inputKnown = true;
            for (Dimension<?> d : inputShape) {
              if (d instanceof NumericDim) {
                inputSize *= ((NumericDim) d).value();
              } else {
                inputKnown = false;
                break;
              }
            }

            List<Dimension<?>> refinedShape = new ArrayList<>(shape);
            if (inputKnown) {
              long inferredDim = inputSize / productKnown;
              refinedShape.set(unknownIndex, new NumericDim((int) inferredDim));
            } else {
              refinedShape.set(unknownIndex, new SymbolicDim("?"));
            }
            refinedShapes.add(refinedShape);
          }
        } else {
          List<Dimension<?>> refinedShape = new ArrayList<>();
          for (Dimension<?> dim : shape) {
            if (dim instanceof NumericDim && ((NumericDim) dim).value() == -1) {
              refinedShape.add(new SymbolicDim("?"));
            } else {
              refinedShape.add(dim);
            }
          }
          refinedShapes.add(refinedShape);
        }
      } else {
        refinedShapes.add(shape);
      }
    }
    return refinedShapes;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Used to resolve `-1` in the target shape — look up the receiver's shape.
    int receiverVn = getReceiverVn();
    if (receiverVn > 0) {
      try {
        return getShapes(builder, getNode(), receiverVn);
      } catch (IllegalArgumentException e) {
        return null;
      }
    }
    return null;
  }

  @Override
  public Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    return getDefaultDTypes(builder);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // `reshape` preserves the receiver's dtype.
    int receiverVn = getReceiverVn();
    if (receiverVn > 0) {
      Set<DType> dtypes = getDTypes(builder, receiverVn);
      if (!dtypes.isEmpty()) return dtypes;
    }
    return Set.of(DType.UNKNOWN);
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE_ARG_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return SHAPE_ARG_NAME;
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
