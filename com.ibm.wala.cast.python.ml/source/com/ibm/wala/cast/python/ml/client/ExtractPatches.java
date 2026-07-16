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
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.image.extract_patches}. Output dtype is inherited from the {@code images}
 * input. Output shape is {@code [batch, out_rows, out_cols, sizes_r * sizes_c * channels]} for the
 * NHWC {@code images}, where {@code out_rows}/{@code out_cols} follow the standard windowed-extent
 * arithmetic over the spatial axes given {@code sizes}, {@code strides}, {@code rates}, and {@code
 * padding}. Derivable when those argument lists and the {@code padding} string are constants and
 * the spatial/channel dimensions of {@code images} are known. See wala/ML#449 (Tier 8).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/image/extract_patches">tf.image.extract_patches</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ExtractPatches extends PassThroughUnaryTensorGenerator {

  public ExtractPatches(PointsToSetVariable source) {
    super(source);
  }

  public ExtractPatches(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "images";
  }

  /**
   * Derives the output shape {@code [batch, out_rows, out_cols, sizes_r * sizes_c * channels]}. The
   * spatial extents follow {@code out = (in - ((size - 1) * rate + 1)) / stride + 1} for {@code
   * VALID} padding and {@code out = ceil(in / stride)} for {@code SAME}. Returns ⊤ unless the
   * {@code sizes}/{@code strides}/{@code rates} lists and the {@code padding} string are constants
   * and the rank-4 {@code images} has known spatial and channel dimensions.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when the shape cannot be
   *     derived.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // images=arg0 (NHWC); sizes=arg1; strides=arg2; rates=arg3; padding=arg4.
    Set<List<Dimension<?>>> imageShapes;
    try {
      imageShapes =
          this.getShapes(builder, this.getArgumentValueNumber(builder, 0, "images", false));
    } catch (IllegalArgumentException e) {
      // wala/ML#584: the `images` operand can't be resolved to a tensor (e.g. a comprehension-built
      // list, whose generator dispatch throws). The result is still a tensor, so degrade to ⊤ shape
      // rather than letting the throw abort the whole type computation and drop the result.
      return null;
    }
    if (imageShapes == null) return null;

    List<Integer> sizes = resolveIntList(builder, 1, "sizes");
    List<Integer> strides = resolveIntList(builder, 2, "strides");
    List<Integer> rates = resolveIntList(builder, 3, "rates");
    String padding = resolveStringArgument(builder, 4, "padding");
    if (sizes == null || strides == null || rates == null || padding == null) return null;
    if (sizes.size() != 4 || strides.size() != 4 || rates.size() != 4) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> image : imageShapes) {
      // extract_patches operates on rank-4 NHWC tensors.
      if (image.size() != 4) continue;
      Dimension<?> rowsDim = image.get(1);
      Dimension<?> colsDim = image.get(2);
      Dimension<?> channelsDim = image.get(3);
      if (!(rowsDim instanceof NumericDim)
          || !(colsDim instanceof NumericDim)
          || !(channelsDim instanceof NumericDim)) continue;

      Integer outRows =
          windowedExtent(
              ((NumericDim) rowsDim).value(), sizes.get(1), strides.get(1), rates.get(1), padding);
      Integer outCols =
          windowedExtent(
              ((NumericDim) colsDim).value(), sizes.get(2), strides.get(2), rates.get(2), padding);
      if (outRows == null || outCols == null) continue;

      int depth = sizes.get(1) * sizes.get(2) * ((NumericDim) channelsDim).value();

      List<Dimension<?>> out = new ArrayList<>();
      out.add(image.get(0)); // batch, carried through (numeric or dynamic)
      out.add(new NumericDim(outRows));
      out.add(new NumericDim(outCols));
      out.add(new NumericDim(depth));
      ret.add(out);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Computes the windowed output extent along one spatial axis.
   *
   * @param in The input extent.
   * @param size The patch size along this axis.
   * @param stride The stride along this axis.
   * @param rate The dilation rate along this axis.
   * @param padding The padding mode ({@code "VALID"} or {@code "SAME"}).
   * @return The output extent, or {@code null} when {@code padding} is unrecognized or {@code
   *     stride} is non-positive.
   */
  private static Integer windowedExtent(int in, int size, int stride, int rate, String padding) {
    if (stride <= 0) return null;
    if (padding.equalsIgnoreCase("VALID")) {
      int effectiveSize = (size - 1) * rate + 1;
      // No window fits when the image is smaller than the (dilated) patch: the output extent is 0,
      // not ⊤. (Java integer division truncates toward zero, so the formula below would
      // incorrectly yield 1 here; hence the explicit 0.) Matches TF, which produces a 0-extent
      // output rather than erroring. See wala/ML#585.
      if (in < effectiveSize) return 0;
      return (in - effectiveSize) / stride + 1;
    }
    if (padding.equalsIgnoreCase("SAME")) {
      return (in + stride - 1) / stride; // ceil(in / stride)
    }
    return null;
  }

  /**
   * Resolves a constant integer-list argument (e.g. {@code sizes=[1, 3, 3, 1]}) to its values.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param position The positional index of the argument.
   * @param name The keyword name of the argument.
   * @return The list of integer values, or {@code null} when the argument is absent, non-constant,
   *     or not a flat list of integers.
   */
  private List<Integer> resolveIntList(
      PropagationCallGraphBuilder builder, int position, String name) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, position, name);
    if (pts == null || pts.isEmpty()) return null;
    Set<List<Dimension<?>>> parsed = this.getShapesFromShapeArgument(builder, pts);
    if (parsed == null || parsed.size() != 1) return null;
    List<Integer> values = new ArrayList<>();
    for (Dimension<?> d : parsed.iterator().next()) {
      if (!(d instanceof NumericDim)) return null;
      values.add(((NumericDim) d).value());
    }
    return values;
  }

  /**
   * Resolves a constant string argument (e.g. {@code padding="VALID"}) to its value.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param position The positional index of the argument.
   * @param name The keyword name of the argument.
   * @return The string value, or {@code null} when the argument is absent, non-constant, or
   *     ambiguous across contexts.
   */
  private String resolveStringArgument(
      PropagationCallGraphBuilder builder, int position, String name) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, position, name);
    if (pts == null || pts.isEmpty()) return null;
    String result = null;
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object val = ((ConstantKey<?>) ik).getValue();
      if (!(val instanceof String)) return null;
      if (result != null && !result.equals(val)) return null;
      result = (String) val;
    }
    return result;
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
