package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for tensors created by {@code tf.data.Dataset.padded_batch}. Like {@link
 * DatasetBatchGenerator}, the batch dimension is prepended; unlike plain {@code batch}, the
 * per-element shape is declared by the {@code padded_shapes} argument (a shape, or a dict/nested
 * structure of shapes, with {@code None} marking dims padded to the longest element), so when that
 * argument resolves it overrides the upstream element shape. See <a
 * href="https://github.com/wala/ML/issues/673">wala/ML#673</a>.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#padded_batch">tf.data.Dataset.padded_batch</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetPaddedBatchGenerator extends DatasetBatchGenerator {

  /** Parameter positions and keyword names for {@code padded_batch}. */
  protected enum PaddedParameters {
    /** The batch size; same position as {@code batch}'s. */
    BATCH_SIZE,

    /** The per-element padded shape structure. */
    PADDED_SHAPES,

    /** The padding values; not consumed by this generator. */
    PADDING_VALUES,

    /** Whether to drop the final partial batch. */
    DROP_REMAINDER;

    /**
     * Lowercase keyword name used in argument-resolution helpers.
     *
     * @return The lowercased enum name (e.g. {@code "padded_shapes"}).
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

  /**
   * Constructs a {@code DatasetPaddedBatchGenerator} from a caller-side {@link
   * PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     padded_batch} invoke.
   */
  public DatasetPaddedBatchGenerator(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getDropRemainderParameterIndex() {
    return PaddedParameters.DROP_REMAINDER.getIndex();
  }

  @Override
  protected String getDropRemainderParameterName() {
    return PaddedParameters.DROP_REMAINDER.getName();
  }

  /**
   * Constructs a {@code DatasetPaddedBatchGenerator} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code padded_batch} synthetic method.
   */
  public DatasetPaddedBatchGenerator(CGNode node) {
    super(node);
  }

  /**
   * Resolves the element shapes from the {@code padded_shapes} argument when it resolves (dims from
   * the declared structure, {@code None} as a dynamic dim), then applies the batch dimension; falls
   * back to the upstream element shape otherwise.
   *
   * @param builder The propagation call graph builder.
   * @return The batched shapes, or {@code null} if neither the argument nor the upstream shape
   *     resolves.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> paddedShapesPts =
        this.getArgumentPointsToSet(
            builder,
            PaddedParameters.PADDED_SHAPES.getIndex(),
            PaddedParameters.PADDED_SHAPES.getName());

    if (paddedShapesPts != null && !paddedShapesPts.isEmpty()) {
      Set<List<Dimension<?>>> elementShapes =
          this.getShapesFromShapeArgument(builder, paddedShapesPts);
      if (elementShapes != null && !elementShapes.isEmpty()) {
        return this.applyBatching(this.padToLongest(elementShapes, builder), builder);
      }
    }

    return super.getDefaultShapes(builder);
  }

  /**
   * Resolves the {@code pad to the longest element in the batch} entries of a {@code padded_shapes}
   * argument (<a href="https://github.com/wala/ML/issues/810">wala/ML#810</a>).
   *
   * <p>A negative entry declares that padding mode rather than an extent. The shape-argument helper
   * is shared with operations where a negative entry means {@code infer this axis}, so it reports
   * the literal number, and taking that literally yields a negative extent: a size no shape can
   * have, and one a specification written from it would be rejected for.
   *
   * <p>Padding to the longest element yields the longest element, so where the upstream element
   * agrees on a definite extent for that axis, that extent is the answer and is tighter than the
   * runtime's own static shape, which reports {@code None} because it does not track the upstream
   * size. Where the upstream extent is itself feed-dependent or unknown, the padded extent varies
   * per batch and {@link DynamicDim} is the honest reading.
   *
   * @param elementShapes The element shapes as read from the argument.
   * @param builder The propagation call graph builder.
   * @return The same shapes with every negative extent resolved.
   */
  private Set<List<Dimension<?>>> padToLongest(
      Set<List<Dimension<?>>> elementShapes, PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> receiverPTS = this.getReceiverPTS(builder);
    Set<List<Dimension<?>>> upstream =
        receiverPTS == null || receiverPTS.isEmpty()
            ? null
            : this.getShapesOfValue(builder, receiverPTS);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    for (List<Dimension<?>> shape : elementShapes) {
      List<Dimension<?>> rewritten = new ArrayList<>(shape.size());

      for (int axis = 0; axis < shape.size(); axis++) {
        Dimension<?> dim = shape.get(axis);

        if (dim instanceof NumericDim && ((NumericDim) dim).value() < 0) {
          Dimension<?> fromUpstream = soleUpstreamDim(upstream, shape.size(), axis);
          rewritten.add(fromUpstream == null ? DynamicDim.INSTANCE : fromUpstream);
        } else rewritten.add(dim);
      }

      ret.add(rewritten);
    }

    return ret;
  }

  /**
   * The upstream element's extent on one axis, when every upstream shape of the expected rank
   * agrees on it (wala/ML#810).
   *
   * @param upstream The upstream element shapes, or {@code null} when they did not resolve.
   * @param rank The rank the {@code padded_shapes} argument declares.
   * @param axis The axis being resolved.
   * @return The agreed extent, or {@code null} when the upstream does not determine one.
   */
  private static Dimension<?> soleUpstreamDim(
      Set<List<Dimension<?>>> upstream, int rank, int axis) {
    if (upstream == null || upstream.isEmpty()) return null;

    Dimension<?> found = null;

    for (List<Dimension<?>> shape : upstream) {
      // A shape of a different rank is not this element, so it determines nothing here.
      if (shape.size() != rank) return null;

      Dimension<?> dim = shape.get(axis);
      // Only a definite extent is tighter than the padding mode's own reading.
      if (!(dim instanceof NumericDim)) return null;
      if (found != null && !found.equals(dim)) return null;
      found = dim;
    }

    return found;
  }
}
