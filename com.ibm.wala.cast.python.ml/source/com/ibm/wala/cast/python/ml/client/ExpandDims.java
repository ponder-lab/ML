package com.ibm.wala.cast.python.ml.client;

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
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.expand_dims(input, axis, name=None)}. Output dtype inherits from {@code
 * input}; output shape is {@code input.shape} with a length-1 dim inserted at {@code axis} (see <a
 * href="https://github.com/wala/ML/issues/500">wala/ML#500</a>).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/expand_dims">tf.expand_dims</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ExpandDims extends PassThroughUnaryTensorGenerator {

  /**
   * Parameter positions and keyword names for {@code tf.expand_dims(input, axis, name=None)}.
   * Ordinals match the position in the XML's {@code paramNames} after the implicit {@code self}
   * receiver.
   */
  protected enum Parameters {
    /** Tensor whose rank is being expanded; shape and dtype source. */
    INPUT,

    /** Position at which to insert the new length-1 dimension; not consumed by this generator. */
    AXIS,

    /** Optional debug name for the op; not consumed by this generator. */
    NAME;

    /**
     * Lowercase keyword name used in arg-resolution helpers when the call site uses {@code
     * keyword=value} syntax.
     *
     * @return The lowercased enum name (e.g. {@code "input"}).
     */
    public String getName() {
      return name().toLowerCase(Locale.ROOT);
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

  public ExpandDims(PointsToSetVariable source) {
    super(source);
  }

  public ExpandDims(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.INPUT.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.INPUT.getName();
  }

  /**
   * Computes the output shape: input shape with a length-1 dim inserted at {@code axis}. For an
   * input of rank {@code r}, {@code axis} is in {@code [-(r+1), r]}; negative values count from the
   * end ({@code -1} means "last position in the output", i.e., a trailing length-1 dim).
   * Per-context input shape unions produce per-context output shapes; an unresolved {@code axis}
   * returns ⊤.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The composed output shapes, one per (input shape, axis value) combination; {@code null}
   *     when the input shape or axis can't be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    int inputVn =
        this.getArgumentValueNumber(
            builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName(), false);
    if (inputVn <= 0) return null;
    Set<List<Dimension<?>>> inputShapes = this.getShapes(builder, inputVn);
    if (inputShapes == null) return null;

    OrdinalSet<InstanceKey> axisPts =
        this.getArgumentPointsToSet(builder, Parameters.AXIS.getIndex(), Parameters.AXIS.getName());
    if (axisPts == null || axisPts.isEmpty()) return null;
    Set<Integer> axisValues = new HashSet<>();
    for (InstanceKey ik : axisPts) {
      Integer axisValue = null;
      if (ik instanceof ConstantKey) {
        Object val = ((ConstantKey<?>) ik).getValue();
        if (val instanceof Number) axisValue = ((Number) val).intValue();
      } else
        // The list form `axis=[-1]` (as in NLPGNN's `WDEmbedding.call`): a single-element
        // list/tuple whose only element is a constant is equivalent to the scalar (wala/ML#714).
        axisValue = singleElementConstant(builder, ik);
      if (axisValue == null) return null;
      axisValues.add(axisValue);
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      int rank = inputShape.size();
      for (int axis : axisValues) {
        int normalized = axis < 0 ? axis + rank + 1 : axis;
        if (normalized < 0 || normalized > rank) continue;
        List<Dimension<?>> newShape = new ArrayList<>(rank + 1);
        for (int i = 0; i < rank + 1; i++) {
          if (i == normalized) newShape.add(new NumericDim(1));
          else if (i < normalized) newShape.add(inputShape.get(i));
          else newShape.add(inputShape.get(i - 1));
        }
        ret.add(newShape);
      }
    }
    return ret.isEmpty() ? null : ret;
  }
}
