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
import java.util.Locale;
import java.util.Set;

/**
 * Base for axis-reduction generators ({@code tf.reduce_*}, {@code tf.argmax}/{@code tf.argmin}):
 * the shared {@code axis}/{@code keepdims} shape collapse. By default the output dtype
 * <em>preserves</em> the input dtype (as {@code tf.reduce_sum}/{@code max}/{@code min} do);
 * subclasses with different dtype semantics override {@link
 * #getDTypes(PropagationCallGraphBuilder)} (e.g. {@link ReduceMean} promotes integers to {@code
 * float32}, {@link Argmax} emits index types). Replaces {@code extends ReduceMean}
 * code-reuse-not-is-a inheritance (<a
 * href="https://github.com/wala/ML/issues/514">wala/ML#514</a>).
 */
public abstract class Reduction extends TensorGenerator {

  protected enum Parameters {
    INPUT_TENSOR,
    AXIS,
    KEEPDIMS,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Reduction(PointsToSetVariable source) {
    super(source);
  }

  /**
   * The positional index of the input-tensor parameter. Overridable so subclasses whose op names
   * the input differently (e.g. {@code argmax}'s {@code input}) resolve keyword calls correctly.
   *
   * @return The zero-based positional index of the input-tensor parameter.
   */
  protected int getInputTensorParameterPosition() {
    return Parameters.INPUT_TENSOR.getIndex();
  }

  /**
   * The keyword name of the input-tensor parameter. Overridable so subclasses whose op names the
   * input differently (e.g. {@code argmax}'s {@code input}) resolve keyword calls correctly.
   *
   * @return The keyword name of the input-tensor parameter.
   */
  protected String getInputTensorParameterName() {
    return Parameters.INPUT_TENSOR.getName();
  }

  /**
   * Resolves the possible {@code keepdims} values for this reduction from the {@code keepdims}
   * argument. When the argument is absent, defaults to {@code {false}}; a non-boolean, non-numeric
   * constant (which can never be a boolean flag) widens to {@code {false, true}} to stay sound.
   * Subclasses whose op has no {@code keepdims} parameter (e.g. {@code argmax}) override this to
   * force {@code {false}}, which also prevents misreading an argument that sits at the same
   * positional index as {@code keepdims} (see {@link Argmax}).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible {@code keepdims} values; never empty.
   */
  protected Set<Boolean> getKeepDimsValues(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> keepDimsPts =
        this.getArgumentPointsToSet(
            builder, Parameters.KEEPDIMS.getIndex(), Parameters.KEEPDIMS.getName());

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
          } else {
            // A non-boolean, non-numeric constant can never be a boolean flag; widen to both for
            // soundness.
            keepDimsValues.add(false);
            keepDimsValues.add(true);
          }
        } else {
          // assume both? or false?
          keepDimsValues.add(false);
          keepDimsValues.add(true);
        }
      }
    }
    if (keepDimsValues.isEmpty()) keepDimsValues.add(false);
    return keepDimsValues;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // If no axis is specified, it reduces all dimensions to a scalar (unless keepdims=True).
    // Default keepdims is False.

    int inputValNum =
        this.getArgumentValueNumber(
            builder,
            this.getInputTensorParameterPosition(),
            this.getInputTensorParameterName(),
            false);
    Set<List<Dimension<?>>> inputShapes = this.getShapes(builder, inputValNum);
    if (inputShapes == null) return null;

    OrdinalSet<InstanceKey> axisPts =
        this.getArgumentPointsToSet(builder, Parameters.AXIS.getIndex(), Parameters.AXIS.getName());

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    Set<Boolean> keepDimsValues = this.getKeepDimsValues(builder);

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
          // Try to handle list/tuple of axes.
          Set<List<Dimension<?>>> axesLists =
              this.getShapesFromShapeArgument(builder, Collections.singleton(ik));
          // The helper has two failure modes: throw `IllegalStateException` for unrecognized
          // top-level allocation types (which propagates up — `performAnalysis` doesn't catch
          // generator exceptions today, so the run aborts), or return `null` for recognized
          // sub-parse failures during recursive descent (empty `tf.constant` value PTS, empty
          // `TensorSpec`/`RaggedTensorSpec` shape PTS, recursive null). Either can fire here
          // when the axis arg is e.g. `tf.constant([...])` whose value-field PTS is empty, so
          // the null guard below is live, not defensive — translate `null` to ⊤ for this
          // generator.
          if (axesLists == null) return null;
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

    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Inherit dtype from the input tensor. Resolve its IR value number first; `getDTypes(builder,
    // int)` expects a value number, not the parameter index. wala/ML#592.
    int inputValNum =
        this.getArgumentValueNumber(
            builder,
            this.getInputTensorParameterPosition(),
            this.getInputTensorParameterName(),
            false);
    return this.getDTypes(builder, inputValNum);
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
    // tf.reduce_mean does not have a 'dtype' parameter that allows specifying the output type
    // directly.
    // The output type is derived from the input type (integers are cast to float32/float64).
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    // A reduction preserves the input tensor's dtype by default (e.g. tf.reduce_sum/max/min/prod);
    // ops with different semantics override this (e.g. ReduceMean promotes integers to float32).
    int inputValNum =
        this.getArgumentValueNumber(
            builder,
            this.getInputTensorParameterPosition(),
            this.getInputTensorParameterName(),
            false);
    return this.getDTypes(builder, inputValNum);
  }
}
