package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
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
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code tf.repeat(input, repeats, axis=None, name=None)}. Output dtype is inherited
 * from {@code input}. With an {@code axis}, the output keeps the input's rank and the extent along
 * {@code axis} becomes the total of the repeats; without one, the input is flattened and the output
 * is rank 1. A scalar {@code repeats} multiplies the extent, and a constant list of repeats sums to
 * it. A tensor {@code repeats}, or a list holding one, gives a {@link DynamicDim}, since
 * TensorFlow's static shape reports {@code None} for a length a runtime tensor determines. Any
 * other {@code repeats} gives an {@link UnresolvedDim}. A supplied {@code axis} not read as a
 * constant may be any of the input's axes, and gives one member per axis; an axis out of the
 * input's range is ⊤.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/repeat">tf.repeat</a>
 */
public class Repeat extends PassThroughUnaryTensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Repeat.class.getName());

  /** Parameter positions and keyword names for {@code tf.repeat(input, repeats, axis, name)}. */
  private enum Parameters {
    INPUT,
    REPEATS,
    AXIS;

    String getName() {
      return name().toLowerCase();
    }

    int getIndex() {
      return ordinal();
    }
  }

  /**
   * Constructs from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public Repeat(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Repeat(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.INPUT.getIndex();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.INPUT.getName();
  }

  /**
   * How the {@code repeats} argument determines a repeated extent: a scalar count, a constant list
   * of counts, a tensor, or something the analysis cannot read.
   *
   * @param scalar The constant scalar count, when {@code repeats} is one.
   * @param counts The constant counts, when {@code repeats} is a constant list.
   * @param tensor Whether {@code repeats} may be a tensor.
   */
  private record Repeats(Integer scalar, List<Integer> counts, boolean tensor) {

    /** The extent that repeating an axis of the given extent gives. */
    Dimension<?> repeat(Dimension<?> extent) {
      if (this.tensor) return DynamicDim.INSTANCE;
      if (this.scalar != null) {
        if (extent instanceof NumericDim numeric)
          return new NumericDim(numeric.value() * this.scalar);
        if (this.scalar == 1) return extent;
        if (this.scalar == 0) return new NumericDim(0);
        return extent instanceof DynamicDim ? DynamicDim.INSTANCE : UnresolvedDim.INSTANCE;
      }
      if (this.counts != null) {
        int total = 0;
        for (int count : this.counts) total += count;
        return new NumericDim(total);
      }
      return UnresolvedDim.INSTANCE;
    }
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null) return null;

    Set<Optional<Integer>> axes = this.resolveAxes(builder);
    Set<Repeats> repeats = this.resolveRepeats(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes) {
      // An axis not read as a constant is each of the input's axes in turn.
      Set<Optional<Integer>> inputAxes = HashSetFactory.make();
      for (Optional<Integer> axis : axes)
        if (axis.equals(ANY_AXIS)) {
          LOGGER.fine(
              () ->
                  "Axis of "
                      + describe(this.getSource())
                      + " not read as a constant; reading each axis of "
                      + input
                      + ".");
          for (int i = 0; i < input.size(); i++) inputAxes.add(Optional.of(i));
        } else inputAxes.add(axis);
      for (Optional<Integer> axis : inputAxes)
        for (Repeats r : repeats) {
          List<Dimension<?>> out = repeatShape(input, axis, r);
          // A ⊤ (null) for any alternative joins to ⊤ for the whole result.
          if (out == null) return null;
          ret.add(out);
        }
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Repeats a single input shape.
   *
   * @param input The input shape.
   * @param axis The axis, or empty to flatten the input first.
   * @param repeats The repeats.
   * @return The output shape, or {@code null} (⊤) when the axis is out of range or the repeated
   *     extent is ragged.
   */
  private static List<Dimension<?>> repeatShape(
      List<Dimension<?>> input, Optional<Integer> axis, Repeats repeats) {
    if (axis.isEmpty()) {
      // The input is flattened: its extent is the product of the input's.
      Dimension<?> size = new NumericDim(1);
      for (Dimension<?> dim : input) {
        if (dim instanceof RaggedDim) return null;
        if (size instanceof NumericDim s && dim instanceof NumericDim d)
          size = new NumericDim(s.value() * d.value());
        else
          size =
              size instanceof DynamicDim || dim instanceof DynamicDim
                  ? DynamicDim.INSTANCE
                  : UnresolvedDim.INSTANCE;
      }
      return List.of(repeats.repeat(size));
    }
    int rank = input.size();
    int normalized = axis.get() < 0 ? axis.get() + rank : axis.get();
    if (normalized < 0 || normalized >= rank) return null;
    Dimension<?> extent = input.get(normalized);
    if (extent instanceof RaggedDim) return null;
    List<Dimension<?>> out = new ArrayList<>(input);
    out.set(normalized, repeats.repeat(extent));
    return out;
  }

  /**
   * Resolves the {@code axis} argument. An omitted argument and a supplied one whose value the
   * analysis cannot read leave the same empty points-to set behind (wala/ML#896); only the omitted
   * one is the default, {@code None}. A supplied axis that is not read as a constant may be any of
   * the input's axes, so it reads as {@link #ANY_AXIS}, and the output keeps the input's rank.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @return The possible axes: empty meaning {@code None}, {@link #ANY_AXIS} meaning an axis not
   *     read as a constant.
   */
  private Set<Optional<Integer>> resolveAxes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(builder, Parameters.AXIS.getIndex(), Parameters.AXIS.getName());
    if (pts == null || pts.isEmpty())
      return Boolean.FALSE.equals(
              this.isArgumentSyntacticallySupplied(
                  builder, Parameters.AXIS.getIndex(), Parameters.AXIS.getName()))
          ? Set.of(Optional.empty())
          : Set.of(ANY_AXIS);
    Set<Optional<Integer>> axes = HashSetFactory.make();
    for (InstanceKey ik : pts) {
      Object value = ik instanceof ConstantKey<?> constant ? constant.getValue() : ANY_AXIS;
      if (value == null) axes.add(Optional.empty());
      else if (value instanceof Number number) axes.add(Optional.of(number.intValue()));
      else axes.add(ANY_AXIS);
    }
    return axes;
  }

  /**
   * The axis an {@code axis} argument stands for when it is supplied but not read as a constant:
   * any of the input's axes. No real axis is this far out of range.
   */
  private static final Optional<Integer> ANY_AXIS = Optional.of(Integer.MIN_VALUE);

  /**
   * Resolves the {@code repeats} argument into one alternative per form its points-to set holds.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @return The alternatives; a single unreadable one when the argument cannot be read.
   */
  private Set<Repeats> resolveRepeats(PropagationCallGraphBuilder builder) {
    Repeats unreadable = new Repeats(null, null, false);
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, Parameters.REPEATS.getIndex(), Parameters.REPEATS.getName());
    if (pts == null || pts.isEmpty()) return Set.of(unreadable);
    Set<Repeats> ret = HashSetFactory.make();
    for (InstanceKey ik : pts) {
      if (ik instanceof ConstantKey<?> constant) {
        ret.add(
            constant.getValue() instanceof Number number
                ? new Repeats(number.intValue(), null, false)
                : unreadable);
        continue;
      }
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin != null && asin.concreteType().getReference().equals(TensorFlowTypes.TENSOR_TYPE)) {
        ret.add(new Repeats(null, null, true));
        continue;
      }
      ret.add(this.countsList(builder, ik).orElse(unreadable));
    }
    return ret;
  }

  /**
   * Reads a list of counts.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param ik The {@code repeats} member.
   * @return Constant counts for a list of constant integers, tensor counts for a list one of whose
   *     elements may be a tensor (read as a {@link DynamicDim}), or empty when the member is
   *     neither.
   */
  private Optional<Repeats> countsList(PropagationCallGraphBuilder builder, InstanceKey ik) {
    Set<List<Dimension<?>>> lists;
    try {
      lists = this.getShapesFromShapeArgument(builder, Collections.singleton(ik));
    } catch (IllegalArgumentException | IllegalStateException e) {
      return Optional.empty();
    }
    if (lists == null || lists.isEmpty()) return Optional.empty();
    for (List<Dimension<?>> list : lists)
      for (Dimension<?> d : list)
        if (d instanceof DynamicDim) return Optional.of(new Repeats(null, null, true));
    if (lists.size() != 1) return Optional.empty();
    List<Integer> counts = new ArrayList<>();
    for (Dimension<?> d : lists.iterator().next()) {
      if (!(d instanceof NumericDim numeric)) return Optional.empty();
      counts.add(numeric.value());
    }
    return Optional.of(new Repeats(null, counts, false));
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes in {@link
   * #getDefaultShapes}, which the pass-through identity record path would bypass.
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
