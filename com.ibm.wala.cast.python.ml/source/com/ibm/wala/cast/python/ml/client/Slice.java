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
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code tf.slice(input_, begin, size, name=None)}. Output dtype is inherited from
 * the {@code input_} input. Output shape is derived per axis from the constant {@code begin}/{@code
 * size} extents and {@code input_.shape}:
 *
 * <pre>
 * output.shape[i] = size[i]                    if size[i] &gt;= 0
 *                 = input_.shape[i] - begin[i] if size[i] == -1  ("all remaining")
 * </pre>
 *
 * A constant {@code size} with no {@code -1} entries gives a fully concrete shape independent of
 * {@code input_.shape}; a {@code -1} entry needs the corresponding {@code input_} dim and {@code
 * begin[i]}, and degrades to a {@link DynamicDim} on that axis (keeping the rank) when either is
 * non-constant. The shape falls back to ⊤ only when {@code begin}/{@code size} are themselves
 * non-constant or their ranks disagree with {@code input_}. See <a
 * href="https://github.com/wala/ML/issues/569">wala/ML#569</a>; dtype forwarding alone landed in <a
 * href="https://github.com/wala/ML/issues/568">wala/ML#568</a>.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/slice">tf.slice</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Slice extends PassThroughUnaryTensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Slice.class.getName());

  public Slice(PointsToSetVariable source) {
    super(source);
  }

  public Slice(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "input_";
  }

  /**
   * Derives the output shape per axis from the constant {@code begin} (arg 1) and {@code size} (arg
   * 2) extents together with the {@code input_} (arg 0) shape, per the rule documented on the
   * class.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when {@code input_}'s shape is
   *     unknown, {@code begin}/{@code size} are non-constant, or their ranks disagree with {@code
   *     input_}.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // input_ is arg 0 (resolved by the passthrough base); begin is arg 1; size is arg 2.
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null) return null;

    Set<List<Dimension<?>>> beginLists = resolveConstantIntList(builder, 1, "begin");
    Set<List<Dimension<?>>> sizeLists = resolveConstantIntList(builder, 2, "size");
    if (beginLists == null || sizeLists == null) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes)
      for (List<Dimension<?>> begin : beginLists)
        for (List<Dimension<?>> size : sizeLists) {
          List<Dimension<?>> out = sliceShape(input, begin, size);
          // A ⊤ (null) for any combination joins to ⊤ for the whole result: returning only the
          // concrete subset would under-approximate the possible shapes.
          if (out == null) return null;
          ret.add(out);
        }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Resolves a {@code begin}/{@code size} argument (a constant list/tuple or {@code tf.constant} of
   * ints) into its dimension lists via {@link #getShapesFromShapeArgument}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param position The 0-based positional index of the argument (excluding {@code self}).
   * @param name The keyword name of the argument.
   * @return The resolved int-list candidates, or {@code null} when the argument's points-to set is
   *     empty or it cannot be resolved to a constant list.
   */
  private Set<List<Dimension<?>>> resolveConstantIntList(
      PropagationCallGraphBuilder builder, int position, String name) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, position, name);
    if (pts == null || pts.isEmpty()) return null;
    try {
      Set<List<Dimension<?>>> lists = this.getShapesFromShapeArgument(builder, pts);
      return (lists == null || lists.isEmpty()) ? null : lists;
    } catch (IllegalStateException e) {
      // `getShapesFromShapeArgument` throws `IllegalStateException` for an unrecognized shape form;
      // degrade that to ⊤. Other runtime exceptions propagate as intended diagnostics.
      LOGGER.fine(
          () ->
              "Could not resolve "
                  + name
                  + " of "
                  + Loggables.describe(this.getSource())
                  + ": "
                  + e
                  + ".");
      return null;
    }
  }

  /**
   * Applies the {@code tf.slice} per-axis shape rule to a single {@code (input, begin, size)}
   * combination.
   *
   * @param input The {@code input_} shape.
   * @param begin The constant {@code begin} offsets.
   * @param size The constant {@code size} extents.
   * @return The output shape, or {@code null} (⊤) when the ranks disagree, a {@code size} entry is
   *     non-constant or invalid ({@code < -1}), or a {@code size}-of-{@code -1} axis computes a
   *     negative extent ({@code begin} past the axis).
   */
  private static List<Dimension<?>> sliceShape(
      List<Dimension<?>> input, List<Dimension<?>> begin, List<Dimension<?>> size) {
    int rank = input.size();
    if (begin.size() != rank || size.size() != rank) return null;

    List<Dimension<?>> out = new ArrayList<>(rank);
    for (int i = 0; i < rank; i++) {
      Dimension<?> sizeDim = size.get(i);
      if (!(sizeDim instanceof NumericDim)) return null; // non-constant size extent.
      int s = ((NumericDim) sizeDim).value();
      if (s >= 0) {
        out.add(new NumericDim(s));
      } else if (s == -1) {
        // "all remaining" along axis i: input_.shape[i] - begin[i], when both are constant.
        Dimension<?> inDim = input.get(i);
        Dimension<?> beginDim = begin.get(i);
        if (inDim instanceof NumericDim && beginDim instanceof NumericDim) {
          int extent = ((NumericDim) inDim).value() - ((NumericDim) beginDim).value();
          // A `begin` past the axis would yield a negative (invalid) extent; degrade to ⊤.
          if (extent < 0) return null;
          out.add(new NumericDim(extent));
        } else out.add(DynamicDim.INSTANCE); // keep the rank; this axis is dynamic.
      } else {
        return null; // size < -1 is invalid for tf.slice.
      }
    }
    return out;
  }
}
