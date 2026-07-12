package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.nn.embedding_lookup}. Output dtype is inherited from the {@code params}
 * argument (the embedding table). Output shape is {@code ids.shape + params.shape[1:]}: each id
 * selects a full row of the embedding table, so the leading {@code ids.shape} indexes the result
 * and the trailing {@code params.shape[1:]} is the per-row embedding. See wala/ML#449 (Tier 8).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup">tf.nn.embedding_lookup</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class EmbeddingLookup extends PassThroughUnaryTensorGenerator {

  public EmbeddingLookup(PointsToSetVariable source) {
    super(source);
  }

  public EmbeddingLookup(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "params";
  }

  /**
   * Derives the output shape as {@code ids.shape + params.shape[1:]}: each id selects a full row of
   * the {@code params} embedding table, so the result is indexed by {@code ids.shape} with each
   * entry being a {@code params.shape[1:]} row.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when either input's shape is
   *     unknown or every {@code params} candidate has rank below 1.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // params is arg 0; ids is arg 1.
    Set<List<Dimension<?>>> paramsShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, 0, "params", false));
    Set<List<Dimension<?>>> idsShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, 1, "ids", false));
    if (paramsShapes == null || idsShapes == null) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> ids : idsShapes) {
      for (List<Dimension<?>> params : paramsShapes) {
        // The embedding table must have at least one axis (the row dimension to index into).
        if (params.isEmpty()) continue;
        List<Dimension<?>> out = new ArrayList<>(ids);
        out.addAll(params.subList(1, params.size()));
        ret.add(out);
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes in {@link
   * #getDefaultShapes}, which the pass-through identity record path would bypass, so the record
   * view routes through the legacy transform until a member-wise upgrade.
   *
   * @param builder The propagation call graph builder.
   * @return The transformed result, with any partial input collapsed by the legacy view.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }
}
