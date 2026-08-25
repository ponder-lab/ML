package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
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

  /** The modeled call's arguments, in summary order ({@code self} excluded). */
  protected enum Parameters {
    PARAMS,
    IDS;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.PARAMS.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.PARAMS.getName();
  }

  /**
   * Legacy view of {@link #getDefaultShapeResult(PropagationCallGraphBuilder)}: a partial's
   * resolvable members stand, per the default mode's contract (wala/ML#716), and only a member-less
   * result collapses to {@code null} (⊤).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when either input resolves no
   *     member or every {@code params} candidate has rank below 1.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    ShapeResult result = this.getDefaultShapeResult(builder);
    return result.isPartial() ? result.members() : result.toLegacy();
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
    // Exact-mode reads, so a partially-resolvable input's remainder marks rather than silently
    // dropping (wala/ML#716, wala/ML#718).
    ShapeResult paramsShapes =
        this.shapeResultOfArgumentValue(
            builder, this.getInputParameterPosition(), this.getInputParameterName());
    ShapeResult idsShapes =
        this.shapeResultOfArgumentValue(
            builder, this.getIndicesParameterPosition(), this.getIndicesParameterName());
    if (paramsShapes.members().isEmpty() || idsShapes.members().isEmpty())
      return ShapeResult.unknown();

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> ids : idsShapes.members()) {
      for (List<Dimension<?>> params : paramsShapes.members()) {
        // A rank-0 table is a guaranteed run-time error for this op, so the skip is
        // infeasible-path pruning (the wala/ML#746 arm-pruning sense), not an unmarked remainder.
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
   * The position of the indices argument ({@code ids} for {@code tf.nn.embedding_lookup}).
   *
   * @return The 0-based positional index of the indices argument.
   */
  protected int getIndicesParameterPosition() {
    return Parameters.IDS.getIndex();
  }

  /**
   * The keyword name of the indices argument: {@code "ids"} for {@code tf.nn.embedding_lookup};
   * {@code Gather} overrides with {@code "indices"}, its summary's name — a keyword-form call
   * resolved under the wrong name would otherwise find no value number.
   *
   * @return The keyword parameter name of the indices argument.
   */
  protected String getIndicesParameterName() {
    return Parameters.IDS.getName();
  }

  /**
   * Resolves one argument's shapes as a record through the value-number pipeline, in exact mode.
   * The vn route is deliberate against the inherited PTS-first {@code shapeResultOfArg}: it alone
   * reaches the φ-arm feasibility walk (wala/ML#746) and the subscript-before-points-to resolution
   * (wala/ML#825), which this generator's indices commonly need.
   *
   * @param builder The propagation call graph builder.
   * @param position The 0-based positional index of the argument.
   * @param name The keyword parameter name.
   * @return The resolution result; ⊤ when the argument's value number cannot be located.
   */
  private ShapeResult shapeResultOfArgumentValue(
      PropagationCallGraphBuilder builder, int position, String name) {
    int vn = this.getArgumentValueNumber(builder, position, name, true);
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
