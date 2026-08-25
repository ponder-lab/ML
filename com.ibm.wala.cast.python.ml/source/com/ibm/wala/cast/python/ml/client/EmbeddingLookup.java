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
   * Legacy view of {@link #getDefaultShapeResult(PropagationCallGraphBuilder)}: the default mode's
   * resolvable subset per wala/ML#716, with a wholly-unresolved result collapsing to {@code null}
   * (⊤) as before.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when neither input resolves any
   *     member or every {@code params} candidate has rank below 1.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder).toLegacy();
  }

  /**
   * Member-wise record view (wala/ML#718; the upgrade the legacy route anticipated, wala/ML#823):
   * the output composes per {@code ids}-member × {@code params}-member pair as {@code ids.shape +
   * params.shape[1:]} — each id selects a full row of the embedding table — and either input's
   * unknown remainder rides through as the result's remainder instead of poisoning the whole set,
   * so a partially resolvable input no longer discards what the other input proves. A wholly
   * unresolved input still yields ⊤: with the ids' rank unknown, no complete output member exists
   * to enumerate, which is the case wala/ML#823's own probe found unpreservable.
   *
   * @param builder The propagation call graph builder.
   * @return The composed result.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    // params is arg 0; ids is arg 1. Exact-mode reads, so a partially-resolvable input's
    // remainder marks rather than silently dropping (wala/ML#716, wala/ML#718).
    ShapeResult paramsShapes = this.shapeResultOfArgument(builder, 0, "params");
    ShapeResult idsShapes = this.shapeResultOfArgument(builder, 1, "ids");
    if (paramsShapes.members().isEmpty() || idsShapes.members().isEmpty())
      return ShapeResult.unknown();

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> ids : idsShapes.members()) {
      for (List<Dimension<?>> params : paramsShapes.members()) {
        // The embedding table must have at least one axis (the row dimension to index into).
        if (params.isEmpty()) continue;
        List<Dimension<?>> out = new ArrayList<>(ids);
        out.addAll(params.subList(1, params.size()));
        ret.add(out);
      }
    }
    return ret.isEmpty()
        ? ShapeResult.unknown()
        : new ShapeResult(ret, paramsShapes.hasUnknown() || idsShapes.hasUnknown());
  }

  /**
   * Routes the output-shape resolution through {@link #getDefaultShapeResult} (this generator has
   * no {@code shape} parameter), so partial results cross the generator boundary instead of
   * collapsing at the base's {@code fromLegacy} lift (wala/ML#718; the MatMul precedent).
   *
   * @param builder The propagation call graph builder.
   * @return The resolution result.
   */
  @Override
  protected ShapeResult getShapeResult(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder);
  }

  /**
   * Resolves one argument's shapes as a record, in exact mode.
   *
   * @param builder The propagation call graph builder.
   * @param position The 0-based positional index of the argument.
   * @param name The keyword parameter name.
   * @return The resolution result; ⊤ when the argument's value number cannot be located.
   */
  private ShapeResult shapeResultOfArgument(
      PropagationCallGraphBuilder builder, int position, String name) {
    int vn = this.getArgumentValueNumber(builder, position, name, false);
    if (vn <= 0) return ShapeResult.unknown();
    return this.getShapeResult(builder, this.getNode(), vn, true);
  }

  /**
   * This generator transforms its input's shape, so forwarding operand shapes would overclaim; the
   * feed carries dtype only (wala/ML#682).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The dtype-only feed over the caller-side input keys, or {@code null} when none is
   *     located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.DTYPE_ONLY);
  }
}
