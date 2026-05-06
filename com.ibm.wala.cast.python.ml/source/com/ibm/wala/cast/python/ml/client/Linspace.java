package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
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
 * length {@code num}; output dtype follows {@code start} (with int → float64 promotion per TF
 * semantics — verified empirically on TF 2.9: {@code tf.linspace(tf.constant(0, dtype=tf.int32),
 * tf.constant(10, dtype=tf.int32), 5).dtype} is {@code float64}, not {@code float32}). The {@code
 * axis} parameter is honored only at its default value of 0 (the rank-1 case); non-default axes
 * return ⊤ shape since they require start/stop to be tensors and the result-shape derivation
 * depends on broadcasting that isn't tracked here.
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

  public Linspace(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The `(num,)` shape is only sound when `axis == 0` (the default rank-1 case). For non-default
    // axes the result rank depends on `start`/`stop` being tensors and on broadcasting we don't
    // track here, so emit ⊤ rather than risk an unsound concrete shape. Detect "axis present"
    // explicitly via `isKeywordArgumentPresent` (for the kwarg form) plus a positional-count check
    // (for `tf.linspace(start, stop, num, name, axis)` positional usage). When axis is absent we
    // proceed with the rank-1 shape; when present we require every PTS instance key to be a
    // `ConstantKey` of value `0`.
    //
    // The earlier `getArgumentValueNumber(..., optional=true) != -1` approach was unsound here:
    // when the source is a `ReturnValueKey` of the inlined `linspace.do` (what `findCreator`
    // typically walks to post-wala/ML#380), `getInvokeInstruction()` returns null and
    // `getArgumentValueNumber` falls through to a manual-node fallback that returns the synthetic
    // method's parameter VN unconditionally, mis-classifying every absent-axis user call as
    // "passed".
    boolean axisPresent =
        this.isKeywordArgumentPresent(builder, Parameters.AXIS.getName())
            || this.getNumberOfPossiblePositionalArguments(builder).stream()
                .anyMatch(n -> n > Parameters.AXIS.getIndex());
    if (axisPresent) {
      OrdinalSet<InstanceKey> axisPts =
          this.getArgumentPointsToSet(
              builder, Parameters.AXIS.getIndex(), Parameters.AXIS.getName());
      if (axisPts == null || axisPts.isEmpty()) return null; // present but unresolved → ⊤
      for (InstanceKey ik : axisPts) {
        if (!(ik instanceof ConstantKey)) return null;
        Object val = ((ConstantKey<?>) ik).getValue();
        if (!(val instanceof Number) || ((Number) val).intValue() != 0) return null;
      }
    }

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
      // tf.linspace promotes integer start/stop to float64 (verified on TF 2.9).
      if (dt == DType.INT32 || dt == DType.INT64) ret.add(DType.FLOAT64);
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
