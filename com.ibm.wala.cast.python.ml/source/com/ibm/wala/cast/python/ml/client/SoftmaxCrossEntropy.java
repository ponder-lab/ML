package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.nn.softmax_cross_entropy_with_logits} and {@code
 * tf.nn.sparse_softmax_cross_entropy_with_logits}. Produces a fresh loss tensor whose shape is
 * {@code logits.shape[:-1]} (the batch axes; the last axis &mdash; the class axis &mdash; is
 * reduced), and whose dtype is always {@code float32} regardless of the label / logits dtype.
 *
 * <p>Previously modeled in {@code tensorflow.xml} as a {@code <return value="labels"/>}
 * pass-through &mdash; semantically wrong, since these ops allocate a new tensor rather than
 * returning the {@code labels} input. Now paired with a {@code <new>+<return>} XML declaration;
 * shape comes from the {@code logits} arg's shape (minus last dim) with the usual summary-local-PTS
 * / caller-walk fallback pattern.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits">tf.nn.softmax_cross_entropy_with_logits</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits">tf.nn.sparse_softmax_cross_entropy_with_logits</a>
 */
public class SoftmaxCrossEntropy extends TensorGenerator {

  /**
   * Positional parameters (after {@code self}) shared by both the sparse and non-sparse variants:
   * {@code labels logits name}.
   */
  private enum Parameters {
    LABELS,
    LOGITS;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  /**
   * Constructs a {@code SoftmaxCrossEntropy} from a caller-side {@link PointsToSetVariable} (the
   * return of the {@code tf.nn.(sparse_)softmax_cross_entropy_with_logits(...)} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public SoftmaxCrossEntropy(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code SoftmaxCrossEntropy} anchored to a manual node. Used by {@link
   * TensorGenerator#createManualGenerator} for the fresh-allocation XML modeling when no
   * caller-side {@link PointsToSetVariable} is available (e.g., the allocation flows through a
   * summary-method return whose PTS is implicit).
   *
   * @param node The {@link CGNode} for the {@code (sparse_)softmax_cross_entropy_with_logits.do()}
   *     synthetic method.
   */
  public SoftmaxCrossEntropy(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output loss-tensor shape. For the sparse variant ({@code
   * sparse_softmax_cross_entropy_with_logits}) the output shape equals {@code labels.shape}
   * directly (labels are 1-D integer class indices). For the non-sparse variant ({@code
   * softmax_cross_entropy_with_logits}), the output is {@code labels.shape[:-1]} / {@code
   * logits.shape[:-1]} (reducing the class axis). Discriminates by the generator's declaring-class
   * type; falls back to {@code logits.shape[:-1]} when the labels arg is unresolvable.
   *
   * @param builder The propagation call graph builder.
   * @return The set of resolved output shapes, or {@code null} if neither arg can be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> labelsShapes =
        shapesOfArg(builder, Parameters.LABELS.getIndex(), Parameters.LABELS.getName());
    if (labelsShapes != null && !labelsShapes.isEmpty() && isSparseVariant()) {
      return labelsShapes;
    }

    Set<List<Dimension<?>>> logitsShapes =
        shapesOfArg(builder, Parameters.LOGITS.getIndex(), Parameters.LOGITS.getName());
    if (logitsShapes == null || logitsShapes.isEmpty()) return labelsShapes;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : logitsShapes) {
      if (!shape.isEmpty()) {
        List<Dimension<?>> reduced = new ArrayList<>(shape);
        reduced.remove(reduced.size() - 1);
        ret.add(reduced);
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Reports whether this generator is modeling the sparse variant (i.e., the declaring class is
   * {@code sparse_softmax_cross_entropy_with_logits}). The sparse variant takes 1-D integer class
   * indices as {@code labels} and emits a loss with {@code labels.shape} directly; the non-sparse
   * variant takes one-hot / soft {@code labels} and reduces the class axis.
   *
   * @return {@code true} iff this is the sparse variant.
   */
  private boolean isSparseVariant() {
    return getNode()
        .getMethod()
        .getDeclaringClass()
        .getReference()
        .equals(
            com.ibm.wala.cast.python.ml.types.TensorFlowTypes
                .SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS
                .getDeclaringClass());
  }

  /**
   * Always returns {@code float32}. Cross-entropy loss is a float regardless of label dtype; even
   * the sparse variant (which takes integer labels) emits a float32 loss tensor.
   *
   * @param builder The propagation call graph builder (unused).
   * @return {@code {FLOAT32\}}.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.FLOAT32);
  }

  /**
   * Resolves shapes of the arg at {@code paramPos}. Tries the summary-local PTS first (via {@link
   * #getShapesOfValue}); on empty, falls back to {@link #getArgumentShapesViaCallers}. Mirrors
   * {@link Sigmoid#shapesOfArg Sigmoid} / {@link MatMul#shapesOfArg MatMul}.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolved shapes, or {@code null} if neither path recovers.
   */
  private Set<List<Dimension<?>>> shapesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, pts);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }
    return this.getArgumentShapesViaCallers(builder, paramPos, paramName);
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
