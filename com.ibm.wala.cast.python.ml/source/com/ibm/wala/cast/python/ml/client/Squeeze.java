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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code tf.squeeze(input, axis=None, name=None)}. Output dtype is inherited from the
 * {@code input} input. Output shape drops singleton axes: when {@code axis} is absent every
 * statically size-1 dimension is removed; when {@code axis} names specific axes, those (which TF
 * requires to be size 1) are removed. Dynamic/symbolic dimensions are never dropped under the
 * {@code axis}-absent form (they are not statically known to be 1). See <a
 * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> (Bucket 2a).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/squeeze">tf.squeeze</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Squeeze extends PassThroughUnaryTensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Squeeze.class.getName());

  public Squeeze(PointsToSetVariable source) {
    super(source);
  }

  public Squeeze(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "input";
  }

  /**
   * Derives the output shape by dropping singleton axes from the {@code input} (arg 0) shape, per
   * the {@code axis} (arg 1) argument.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when {@code input}'s shape is
   *     unknown or {@code axis} is a non-constant.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // input is the passthrough base's arg 0; axis is arg 1 (default None -> squeeze all).
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null) return null;

    OrdinalSet<InstanceKey> axisPts = this.getArgumentPointsToSet(builder, 1, "axis");
    Set<Integer> axes; // null means "squeeze every statically size-1 axis".
    if (isAbsentOrNone(axisPts)) {
      axes = null;
    } else {
      axes = resolveAxisInts(builder, axisPts);
      if (axes == null) {
        LOGGER.fine(() -> "Non-constant axis for " + this.getSource() + "; returning ⊤.");
        return null;
      }
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes) {
      List<Dimension<?>> out = squeezeShape(input, axes);
      if (out != null) ret.add(out);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Whether the {@code axis} argument is absent or {@code None} (i.e. squeeze every size-1 axis).
   *
   * @param axisPts The {@code axis} argument's points-to set.
   * @return {@code true} iff {@code axis} is absent ({@code null}/empty PTS) or every element is
   *     the {@code None} constant.
   */
  private static boolean isAbsentOrNone(OrdinalSet<InstanceKey> axisPts) {
    if (axisPts == null || axisPts.isEmpty()) return true;
    for (InstanceKey ik : axisPts) {
      if (!(ik instanceof ConstantKey) || ((ConstantKey<?>) ik).getValue() != null) return false;
    }
    return true;
  }

  /**
   * Resolves the {@code axis} argument to its constant integer axes, accepting a single integer or
   * a constant list/tuple of integers.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param axisPts The {@code axis} argument's points-to set.
   * @return The set of axes, or {@code null} when any element is non-constant.
   */
  private Set<Integer> resolveAxisInts(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> axisPts) {
    Set<Integer> axes = new HashSet<>();
    for (InstanceKey ik : axisPts) {
      if (ik instanceof ConstantKey) {
        Object v = ((ConstantKey<?>) ik).getValue();
        if (v instanceof Number) axes.add(((Number) v).intValue());
        else return null;
      } else {
        // A list/tuple of axes: resolve its elements via the shape-argument parser.
        Set<List<Dimension<?>>> lists;
        try {
          lists = this.getShapesFromShapeArgument(builder, Collections.singleton(ik));
        } catch (RuntimeException e) {
          return null;
        }
        if (lists == null) return null;
        for (List<Dimension<?>> list : lists)
          for (Dimension<?> d : list) {
            if (!(d instanceof NumericDim)) return null;
            axes.add(((NumericDim) d).value());
          }
      }
    }
    return axes;
  }

  /**
   * Drops singleton axes from one input shape. With {@code axes == null} every statically size-1
   * dimension is removed; otherwise the named axes are removed (negative axes count from the end).
   *
   * @param input The {@code input} shape.
   * @param axes The axes to drop, or {@code null} to drop all statically size-1 axes.
   * @return The squeezed shape, or {@code null} when a named axis is a known non-1 dimension (an
   *     invalid {@code tf.squeeze} on a non-singleton axis).
   */
  private static List<Dimension<?>> squeezeShape(List<Dimension<?>> input, Set<Integer> axes) {
    if (axes == null) {
      List<Dimension<?>> out = new ArrayList<>(input.size());
      for (Dimension<?> d : input)
        if (!(d instanceof NumericDim && ((NumericDim) d).value() == 1)) out.add(d);
      return out;
    }
    int rank = input.size();
    Set<Integer> normalized = new HashSet<>();
    for (int a : axes) normalized.add(a < 0 ? a + rank : a);
    List<Dimension<?>> out = new ArrayList<>(input.size());
    for (int i = 0; i < rank; i++) {
      if (normalized.contains(i)) {
        Dimension<?> d = input.get(i);
        if (d instanceof NumericDim && ((NumericDim) d).value() != 1)
          return null; // not squeezable.
        // Else drop this axis.
      } else {
        out.add(input.get(i));
      }
    }
    return out;
  }
}
