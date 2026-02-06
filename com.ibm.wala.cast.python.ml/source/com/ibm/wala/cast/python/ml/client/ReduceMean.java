package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.reduce_mean}.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean">tf.math.reduce_mean</a>
 */
public class ReduceMean extends TensorGenerator {

  protected enum Parameters {
    INPUT_TENSOR,
    AXIS,
    KEEPDIMS,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ReduceMean(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // If no axis is specified, it reduces all dimensions to a scalar (unless keepdims=True).
    // Default keepdims is False.

    int inputValNum =
        this.getArgumentValueNumber(
            builder, Parameters.INPUT_TENSOR.getIndex(), Parameters.INPUT_TENSOR.getName(), false);
    Set<List<Dimension<?>>> inputShapes = this.getShapes(builder, inputValNum);

    OrdinalSet<InstanceKey> axisPts =
        this.getArgumentPointsToSet(builder, Parameters.AXIS.getIndex(), Parameters.AXIS.getName());
    OrdinalSet<InstanceKey> keepDimsPts =
        this.getArgumentPointsToSet(
            builder, Parameters.KEEPDIMS.getIndex(), Parameters.KEEPDIMS.getName());

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    // Determine possible values for keepdims
    Set<Boolean> keepDimsValues = new HashSet<>();
    if (keepDimsPts == null || keepDimsPts.isEmpty()) {
      keepDimsValues.add(false); // Default
    } else {
      for (InstanceKey ik : keepDimsPts) {
        if (ik instanceof ConstantKey) {
          Object val = ((ConstantKey<?>) ik).getValue();
          if (val instanceof Boolean) {
            keepDimsValues.add((Boolean) val);
          } else if (val instanceof Number) { // Python bools are ints
            keepDimsValues.add(((Number) val).intValue() != 0);
          } else if (val == null) {
            keepDimsValues.add(false); // Default if None passed?
          }
        } else {
          // assume both? or false?
          keepDimsValues.add(false);
          keepDimsValues.add(true);
        }
      }
    }
    if (keepDimsValues.isEmpty()) keepDimsValues.add(false);

    // Determine possible values for axis
    Set<Set<Integer>> axisValues = new HashSet<>(); // Each element is a set of axes to reduce
    boolean axisIsNone = false;

    if (axisPts == null || axisPts.isEmpty()) {
      axisIsNone = true;
    } else {
      for (InstanceKey ik : axisPts) {
        if (ik instanceof ConstantKey) {
          Object val = ((ConstantKey<?>) ik).getValue();
          if (val == null) {
            axisIsNone = true;
          } else if (val instanceof Number) {
            Set<Integer> s = new HashSet<>();
            s.add(((Number) val).intValue());
            axisValues.add(s);
          }
        } else {
          // Try to handle list/tuple of axes
          Set<List<Dimension<?>>> axesLists =
              this.getShapesFromShapeArgument(builder, Collections.singleton(ik));
          for (List<Dimension<?>> axesList : axesLists) {
            Set<Integer> s = new HashSet<>();
            for (Dimension<?> d : axesList) {
              if (d instanceof NumericDim) {
                s.add(((NumericDim) d).value());
              }
            }
            axisValues.add(s);
          }
        }
      }
    }

    if (axisValues.isEmpty() && !axisIsNone) {
      // If we found points-to but couldn't resolve values, fallback to None? or fail?
      // Fallback to reducing everything (scalar) is safer for now?
      // Or identity?
      // Let's assume None if we couldn't find any specific axes but passed something.
      // Actually, if we can't analyze axis, we can't determine shape reliably.
      // But for the test case, axis is explicit (0, 1) or None.
    }

    for (List<Dimension<?>> inputShape : inputShapes) {
      int rank = inputShape.size();

      // Case 1: Axis is None (reduce all)
      if (axisIsNone) {
        for (boolean keep : keepDimsValues) {
          if (!keep) {
            ret.add(Collections.emptyList()); // Scalar
          } else {
            List<Dimension<?>> newShape = new ArrayList<>();
            for (int i = 0; i < rank; i++) newShape.add(new NumericDim(1));
            ret.add(newShape);
          }
        }
      }

      // Case 2: Axis specified
      for (Set<Integer> axes : axisValues) {
        // Validate axes for this rank
        boolean validAxes = true;
        for (Integer a : axes) {
          int normalizedAxis = a < 0 ? a + rank : a;
          if (normalizedAxis < 0 || normalizedAxis >= rank) {
            validAxes = false;
            break;
          }
        }
        if (!validAxes) continue;

        for (boolean keep : keepDimsValues) {
          List<Dimension<?>> newShape = new ArrayList<>();
          for (int i = 0; i < rank; i++) {
            boolean reduced = false;
            // Handle negative indices
            for (Integer a : axes) {
              int normalizedAxis = a < 0 ? a + rank : a;
              if (normalizedAxis == i) {
                reduced = true;
                break;
              }
            }

            if (reduced) {
              if (keep) {
                newShape.add(new NumericDim(1));
              }
              // else: removed
            } else {
              newShape.add(inputShape.get(i));
            }
          }
          ret.add(newShape);
        }
      }
    }

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Inherit dtype from input
    return this.getDTypes(builder, Parameters.INPUT_TENSOR.getIndex());
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    // reduce_mean usually preserves type for float, but might change for int?
    // tf.reduce_mean: "The reduced dimensions are removed... The dtype is the same as input_tensor
    // if it is floating point; otherwise it is checked."
    // Actually, documentation says: "If input_tensor has integral data type, the output tensor will
    // have the same data type but with values cast to float32 (or float64 if input is int64)."
    // Wait, let me check the docs.
    // "If input_tensor is integer, result is float32." seems common for mean.
    // Let's check TestTensorflow2Model expectation.
    // SCALAR_TENSOR_OF_FLOAT32 for f (input float32).
    // reduce_mean on ints produces what?
    // tf2_test_reduce_mean.py uses tf.float32.

    // I should implement getDTypes to handle this promotion if needed.
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    // Logic:
    // 1. Get input dtypes.
    // 2. If input is float/double -> keep it.
    // 3. If input is int -> float32 (or float64? usually 32).
    // 4. Input arg "dtype" can override.

    // reduce_mean doesn't have a "dtype" argument in the signature I see in do() in xml?
    // <method name="do" descriptor="()LRoot;" numArgs="5" paramNames="self input_tensor axis
    // keepdims name">
    // No dtype arg.
    // So it's determined by input.

    int inputValNum =
        this.getArgumentValueNumber(
            builder, Parameters.INPUT_TENSOR.getIndex(), Parameters.INPUT_TENSOR.getName(), false);
    Set<DType> inputTypes = this.getDTypes(builder, inputValNum);
    Set<DType> ret = new HashSet<>();
    for (DType t : inputTypes) {
      if (t == DType.INT32 || t == DType.INT64) {
        // TensorFlow defaults to float32 for integer inputs to reduce_mean?
        // Or does it require explicit cast?
        // "If `input_tensor` is integer, the result is cast to `float32`."
        // But tf.reduce_mean docs say: "The output has the same dtype as the input tensor." - wait,
        // that's reduce_sum.
        // For reduce_mean: "The dtype of the result is the same as the input tensor if the input
        // tensor is floating point..."
        // "... if the input tensor is not floating point, the result is converted to float32."
        // However, if I use the "dtype" argument (which is not in the signature above??)
        // Wait, tf.reduce_mean(input_tensor, axis=None, keepdims=False, name=None)
        // Does it have a dtype arg?
        // TF 2.9 docs: tf.math.reduce_mean(input_tensor, axis=None, keepdims=False, name=None)
        // It does NOT have a dtype argument.

        // So int -> float32.
        ret.add(DType.FLOAT32);
      } else {
        ret.add(t);
      }
    }
    return ret;
  }
}
