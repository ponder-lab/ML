package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
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

    /** Whether to drop the final partial batch; not consumed by this generator. */
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
        return this.applyBatching(elementShapes, builder);
      }
    }

    return super.getDefaultShapes(builder);
  }
}
