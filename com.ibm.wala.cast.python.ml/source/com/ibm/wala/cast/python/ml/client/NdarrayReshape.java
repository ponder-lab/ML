package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
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
   * Syntactic check: is {@code source} the result of a method-style {@code x.reshape(shape)} call?
   * Mirrors {@link NdarraySubscriptOperation#isApplicable} — used to dispatch this generator
   * without relying on PTS (which is lost through tuple unpacking chains like {@code x_train,
   * x_test = x_train.astype(...), x_test.astype(...)}).
   *
   * <p>Discriminates from {@code tf.reshape(x, shape)} (a function call with 2 tensor arguments) by
   * requiring the invoke to have exactly 2 uses (function object + shape), matching the method-call
   * ABI where the tensor is the captured receiver rather than an explicit arg.
   *
   * @param source The {@link PointsToSetVariable} whose defining invoke instruction is being
   *     inspected.
   * @param builder The propagation call graph builder used to resolve the function-object property
   *     read's member ref.
   * @return {@code true} iff {@code source}'s def is a 2-use {@link PythonInvokeInstruction} whose
   *     function object is a {@link PythonPropertyRead} with the constant-string member {@code
   *     "reshape"}.
   */
  public static boolean isApplicable(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    if (!(source.getPointerKey() instanceof LocalPointerKey)) return false;
    LocalPointerKey lpk = (LocalPointerKey) source.getPointerKey();
    CGNode node = lpk.getNode();
    SSAInstruction def = node.getDU().getDef(lpk.getValueNumber());
    if (!(def instanceof PythonInvokeInstruction)) return false;
    PythonInvokeInstruction call = (PythonInvokeInstruction) def;
    // `x.reshape(shape)` → 2 uses (func object, shape). `tf.reshape(x, shape)` → 3 uses.
    if (call.getNumberOfUses() != 2) return false;
    SSAInstruction funcDef = node.getDU().getDef(call.getUse(0));
    if (!(funcDef instanceof PythonPropertyRead)) return false;
    PythonPropertyRead propRead = (PythonPropertyRead) funcDef;
    PointerKey memberKey =
        builder
            .getPointerAnalysis()
            .getHeapModel()
            .getPointerKeyForLocal(node, propRead.getMemberRef());
    for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(memberKey)) {
      if (!(ik instanceof ConstantKey)) continue;
      Object value = ((ConstantKey<?>) ik).getValue();
      if ("reshape".equals(value)) {
        LOGGER.fine(() -> "NdarrayReshape.isApplicable: matched for source=" + source);
        return true;
      }
    }
    return false;
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
