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
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.linspace(start, stop, num, name=None, axis=0)}. Output is a 1-D tensor of
 * length {@code num}; output dtype follows {@code start} (with int → float32 promotion per TF
 * semantics: integer start/stop produce a float32 result). The {@code axis} parameter is honored
 * only at its default value of 0 (the rank-1 case); non-default axes return ⊤ shape since they
 * require start/stop to be tensors and the result-shape derivation depends on broadcasting that
 * isn't tracked here.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/linspace">tf.linspace</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Linspace extends TensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.linspace(start, stop, num, name=None,
   * axis=0)}. Ordinals match the position in the XML's {@code paramNames} after the implicit {@code
   * self} receiver, so {@code Parameters.START.getIndex() == 0} resolves to the first user-facing
   * positional argument.
   */
  protected enum Parameters {
    /** The starting value of the sequence. The output dtype follows {@code start}'s dtype. */
    START,

    /** The end value of the sequence (inclusive). */
    STOP,

    /** Number of values in the sequence; determines the output's leading dimension. */
    NUM,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME,

    /**
     * Axis along which the sequence is generated. Honored only at its default value of 0; non-zero
     * axes require {@code start}/{@code stop} to be tensors and produce ⊤ shape.
     */
    AXIS;

    /**
     * Lowercase keyword name used in {@link #getArgumentPointsToSet} / similar arg-resolution
     * helpers when the call site uses {@code keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "start"}).
     */
    public String getName() {
      return name().toLowerCase();
    }

    /**
     * Positional index of this parameter, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based positional index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public Linspace(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> numPts =
        this.getArgumentPointsToSet(builder, Parameters.NUM.getIndex(), Parameters.NUM.getName());
    if (numPts == null || numPts.isEmpty()) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (InstanceKey ik : numPts) {
      if (ik instanceof ConstantKey) {
        Object val = ((ConstantKey<?>) ik).getValue();
        if (val instanceof Number) {
          List<Dimension<?>> shape = new ArrayList<>(1);
          shape.add(new NumericDim(((Number) val).intValue()));
          ret.add(shape);
        }
      }
    }
    // Lattice convention: a `null` return signals ⊤ ("tensor of unknown shape"), while an empty
    // set signals ⊥ ("not a tensor"). When `num`'s PTS contains only non-numeric or non-Constant
    // keys we recovered no concrete shape, but we still know the call returns a tensor — so emit
    // ⊤ rather than ⊥. See `TensorGenerator`'s class-level Javadoc and CONTRIBUTING.md's "Tensor
    // Type Generators" section.
    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int startVn =
        this.getArgumentValueNumber(
            builder, Parameters.START.getIndex(), Parameters.START.getName(), false);
    Set<DType> startDTypes = this.getDTypes(builder, startVn);
    if (startDTypes == null || startDTypes.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = new HashSet<>();
    for (DType dt : startDTypes) {
      // tf.linspace promotes integer start/stop to float32.
      if (dt == DType.INT32 || dt == DType.INT64) ret.add(DType.FLOAT32);
      else ret.add(dt);
    }
    return ret;
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
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
