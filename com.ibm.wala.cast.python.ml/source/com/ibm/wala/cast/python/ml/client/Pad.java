package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.pad(tensor, paddings, mode='CONSTANT', constant_values=0, name=None)}.
 * The output has the input's rank and dtype; each extent grows by the two widths of the matching
 * row of {@code paddings}, an {@code [r, 2]} list of {@code [before, after]} pairs (wala/ML#1009).
 *
 * <p>A row whose widths are not integer constants leaves its extent {@link UnresolvedDim}, a fixed
 * size the analysis did not compute; a {@link DynamicDim} input extent stays {@code Dynamic}, since
 * padding a feed-dependent axis leaves it feed-dependent. A {@code paddings} of a different rank
 * than the input is not this operation's input and contributes nothing. An input of unknown rank
 * takes the rank of the {@code paddings} literal, one axis per row.
 *
 * @see <a href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/pad">tf.pad</a>
 */
public class Pad extends PassThroughUnaryTensorGenerator {

  /** The position of {@code paddings} among the call's arguments. */
  private static final int PADDINGS_POSITION = 1;

  /** The keyword name of {@code paddings}. */
  private static final String PADDINGS_NAME = "paddings";

  public Pad(PointsToSetVariable source) {
    super(source);
  }

  public Pad(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "tensor";
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    OrdinalSet<InstanceKey> paddingsPts =
        this.getArgumentPointsToSet(builder, PADDINGS_POSITION, PADDINGS_NAME);
    // An input of unknown rank still has the rank `paddings` gives, one row per axis; each extent
    // is the unknown input extent grown by its row, which the analysis did not compute.
    if (inputShapes == null) return unknownRankInputShapes(builder, paddingsPts);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes) {
      if (input == null) {
        Set<List<Dimension<?>>> byRows = unknownRankInputShapes(builder, paddingsPts);
        if (byRows == null) return null;
        ret.addAll(byRows);
        continue;
      }
      Set<List<Long[]>> rowsets = paddingRows(builder, paddingsPts, input.size());
      if (rowsets.isEmpty()) {
        // No paddings of this rank resolved: the rank stands, every non-dynamic extent is a fixed
        // size the analysis did not compute.
        List<Dimension<?>> out = new ArrayList<>();
        for (Dimension<?> extent : input)
          out.add(extent instanceof DynamicDim ? extent : UnresolvedDim.INSTANCE);
        ret.add(out);
        continue;
      }
      for (List<Long[]> rows : rowsets) {
        List<Dimension<?>> out = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) out.add(padded(input.get(i), rows.get(i)));
        ret.add(out);
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * The result shapes for an input of unknown rank: {@code tf.pad} requires one {@code [before,
   * after]} row of {@code paddings} per input axis, so each {@code paddings} literal's row count is
   * the result's rank, every extent {@link UnresolvedDim} (wala/ML#1009).
   *
   * @param builder The propagation call graph builder.
   * @param paddingsPts The {@code paddings} argument's points-to set.
   * @return One shape per row count, or {@code null} when no {@code paddings} literal is resolved.
   */
  private static Set<List<Dimension<?>>> unknownRankInputShapes(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> paddingsPts) {
    if (paddingsPts == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (InstanceKey outer : paddingsPts) {
      if (!(outer instanceof AllocationSiteInNode outerSite)) return null;
      int rows =
          integerCatalogSize(
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(outerSite)));
      // A `paddings` with no indexed row is a tensor or an unread container, whose row count is
      // not known here.
      if (rows == 0) return null;
      ret.add(new ArrayList<>(Collections.nCopies(rows, UnresolvedDim.INSTANCE)));
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Grows one extent by a row's two widths.
   *
   * @param extent The input extent.
   * @param row The {@code [before, after]} widths, either {@code null} when not constant.
   * @return The padded extent.
   */
  private static Dimension<?> padded(Dimension<?> extent, Long[] row) {
    if (extent instanceof DynamicDim) return extent;
    if (extent instanceof NumericDim numeric && row[0] != null && row[1] != null)
      return new NumericDim(numeric.value() + row[0].intValue() + row[1].intValue());
    return UnresolvedDim.INSTANCE;
  }

  /**
   * The {@code paddings} alternatives of the given rank, each a list of {@code [before, after]}
   * rows read from the list or tuple literal's constant elements.
   *
   * @param builder The propagation call graph builder.
   * @param paddingsPts The {@code paddings} argument's points-to set.
   * @param rank The input's rank.
   * @return The alternatives; a row's width is {@code null} when it is not an integer constant.
   */
  private static Set<List<Long[]>> paddingRows(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> paddingsPts, int rank) {
    Set<List<Long[]>> ret = HashSetFactory.make();
    if (paddingsPts == null) return ret;
    for (InstanceKey outer : paddingsPts) {
      if (!(outer instanceof AllocationSiteInNode outerSite)) continue;
      List<Long[]> rows = new ArrayList<>();
      for (int i = 0; i < rank; i++) {
        OrdinalSet<InstanceKey> rowPts =
            getInstanceFieldPointsToSet(builder, outerSite, Integer.toString(i));
        Long[] row = {null, null};
        if (rowPts != null && rowPts.size() == 1) {
          InstanceKey inner = rowPts.iterator().next();
          if (inner instanceof AllocationSiteInNode innerSite)
            for (int j = 0; j < 2; j++) row[j] = constantLong(builder, innerSite, j);
        }
        rows.add(row);
      }
      ret.add(rows);
    }
    return ret;
  }

  /**
   * The integer constant stored at an index of a list or tuple literal.
   *
   * @param builder The propagation call graph builder.
   * @param literal The literal.
   * @param index The index.
   * @return The constant, or {@code null} when the element is not a single integer constant.
   */
  private static Long constantLong(
      PropagationCallGraphBuilder builder, AllocationSiteInNode literal, int index) {
    OrdinalSet<InstanceKey> pts =
        getInstanceFieldPointsToSet(builder, literal, Integer.toString(index));
    if (pts == null || pts.size() != 1) return null;
    InstanceKey key = pts.iterator().next();
    if (key instanceof ConstantKey<?> constant && constant.getValue() instanceof Number number)
      return number.longValue();
    return null;
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes, which the
   * pass-through identity record path would bypass.
   *
   * @param builder The propagation call graph builder.
   * @return The transformed result, with any partial input collapsed by the legacy view.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  /**
   * This generator transforms its input's shape, so the feed carries dtype only (wala/ML#682).
   *
   * @param builder The propagation call graph builder.
   * @return The dtype-only feed over the caller-side input keys, or {@code null} when none is
   *     located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.DTYPE_ONLY);
  }
}
