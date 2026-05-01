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
 * product of known dims, matching {@link Reshape}'s algorithm.
 *
 * <p>Dispatch is class-type, keyed on {@code NumpyTypes.RESHAPE_METHOD.getDeclaringClass()}. The
 * call graph resolves the callee via the summary-populated receiver fields (see {@code numpy.xml}'s
 * {@code astype} / {@code reshape} classes). Disjoint from {@code tf.reshape} by virtue of distinct
 * declaring classes ({@code Lnumpy/ndarray/reshape} vs {@code Ltensorflow/functions/reshape}).
 *
 * <p>A former syntactic {@code isApplicable} predicate (mirror of {@link
 * NdarraySubscriptOperation#isApplicable}) was removed as part of wala/ML#402 cleanup after the
 * XML-declaration audit landed &mdash; with `<new>` target classes now correctly registered, the
 * receiver's PTS is concrete and class-type dispatch resolves the callee without needing a fallback
 * syntactic probe.
 */
public class NdarrayReshape extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(NdarrayReshape.class.getName());

  /**
   * Positional parameters of the {@code ndarray.reshape(shape)} invocation. The receiver (the
   * tensor being reshaped) is captured in the function's closure, not an explicit argument, so only
   * the target-shape argument is listed here.
   */
  private enum Parameters {
    /**
     * The target shape. A list/tuple of integers; may contain {@code -1} for a single inferred dim.
     */
    SHAPE;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public NdarrayReshape(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Resolves the value number of the receiver (the tensor being reshaped). For method-style calls
   * {@code x.reshape(shape)}, the receiver is captured in the function-object's closure rather than
   * appearing as an explicit argument. We read it from the {@link PythonPropertyRead} whose member
   * is {@code reshape}: the property read's {@code objectRef} is the receiver's value number in the
   * caller's IR. Mirrors the pattern in {@link AstypeOperation#getReceiverVn}.
   *
   * @return The SSA value number of the receiver in the containing node, or {@code -1} if the
   *     invoke's function object isn't a property read (defensive fallback).
   */
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
        this.getArgumentPointsToSet(
            builder, Parameters.SHAPE.getIndex(), Parameters.SHAPE.getName());
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
    final Set<List<Dimension<?>>> finalShapes = refinedShapes;
    LOGGER.fine(() -> "NdarrayReshape.getShapes: final refined=" + finalShapes);
    return refinedShapes;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Used to resolve `-1` in the target shape — look up the receiver's shape. Uses the
    // PTS-first-then-SSA-DU helper inherited from `TensorGenerator`, which handles the
    // implicit-PK case (wala/WALA#1889) without materialising PTS. IAE from the helper means
    // "empty PTS and SSA walk didn't recognise the creator"; for the reshape receiver that just
    // means we can't resolve `-1`, not that the receiver isn't a tensor — the caller in {@link
    // #getShapes} is expected to fall back to a partial shape (SymbolicDim for the `-1` slot).
    int receiverVn = getReceiverVn();
    if (receiverVn <= 0) return null;
    try {
      return getShapesOrSSAChain(builder, getNode(), receiverVn);
    } catch (IllegalArgumentException e) {
      LOGGER.fine("NdarrayReshape.getDefaultShapes: IAE on receiver vn=" + receiverVn);
      return null;
    }
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
    return Parameters.SHAPE.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
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
