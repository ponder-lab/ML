package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;

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
 * Generator for {@code tf.transpose(a, perm=None, conjugate=False, name='transpose')}. Output dtype
 * is inherited from the {@code a} input. Output shape permutes the input axes by {@code perm}: when
 * {@code perm} is absent or {@code None}, the axes are reversed (the default transpose); when
 * {@code perm} is a constant permutation, axis {@code i} of the output is axis {@code perm[i]} of
 * the input. A non-constant {@code perm}, or a {@code perm} that is not a valid permutation of the
 * input's axes, falls back to ⊤ rather than risk an unsound shape. Previously modeled as a
 * first-argument {@code pass_through}, which reported the input shape unchanged. See <a
 * href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/transpose">tf.transpose</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Transpose extends PassThroughUnaryTensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Transpose.class.getName());

  /**
   * Constructs from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public Transpose(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Transpose(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "a";
  }

  /**
   * Derives the output shape by permuting the {@code a} (arg 0) shape per the {@code perm} (arg 1)
   * argument: reversed when {@code perm} is absent or {@code None}, otherwise reordered so output
   * axis {@code i} is input axis {@code perm[i]}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when {@code a}'s shape is
   *     unknown, {@code perm} is a non-constant, or {@code perm} is not a valid permutation of the
   *     input's axes for some input alternative.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder).toLegacy();
  }

  /**
   * Member-wise record view (wala/ML#718): the permutation applies per resolvable input member, and
   * a member the permutation cannot apply to joins the unknown remainder instead of collapsing the
   * whole result.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The permuted result.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    ShapeResult input = super.getDefaultShapeResult(builder);
    if (input.members().isEmpty()) return input;

    OrdinalSet<InstanceKey> permPts = this.getArgumentPointsToSet(builder, 1, "perm");
    List<Integer> perm; // null means "reverse every axis" (the default transpose).
    if (isAbsentOrNone(permPts)) {
      perm = null;
    } else {
      perm = resolvePermList(builder, permPts);
      if (perm == null) {
        LOGGER.fine(() -> "Non-constant perm for " + describe(this.getSource()) + "; returning ⊤.");
        return ShapeResult.unknown();
      }
    }

    boolean hasUnknown = input.hasUnknown();
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> in : input.members()) {
      List<Dimension<?>> out = permuteShape(in, perm);
      // A member the permutation cannot apply to joins the unknown remainder.
      if (out == null) hasUnknown = true;
      else ret.add(out);
    }
    return ret.isEmpty() ? ShapeResult.unknown() : new ShapeResult(ret, hasUnknown);
  }

  /**
   * Permutes a single input shape. With a {@code null} permutation the axes are reversed; otherwise
   * output axis {@code i} takes input axis {@code perm[i]}.
   *
   * @param input The input shape.
   * @param perm The permutation, or {@code null} to reverse all axes.
   * @return The permuted shape, or {@code null} (⊤) when {@code perm} is not a valid permutation of
   *     {@code input}'s axes.
   */
  private static List<Dimension<?>> permuteShape(List<Dimension<?>> input, List<Integer> perm) {
    if (perm == null) {
      List<Dimension<?>> reversed = new ArrayList<>(input);
      Collections.reverse(reversed);
      return reversed;
    }
    // perm must be a permutation of exactly the input's axes; anything else is unsound.
    if (perm.size() != input.size()) return null;
    Set<Integer> seen = new HashSet<>();
    for (int axis : perm) {
      if (axis < 0 || axis >= input.size() || !seen.add(axis)) return null;
    }
    List<Dimension<?>> out = new ArrayList<>(input.size());
    for (int axis : perm) out.add(input.get(axis));
    return out;
  }

  /**
   * Whether the {@code perm} argument is absent or {@code None} (i.e. reverse every axis).
   *
   * @param permPts The {@code perm} argument's points-to set.
   * @return {@code true} iff {@code perm} is absent ({@code null}/empty PTS) or every element is
   *     the {@code None} constant.
   */
  private static boolean isAbsentOrNone(OrdinalSet<InstanceKey> permPts) {
    if (permPts == null || permPts.isEmpty()) return true;
    for (InstanceKey ik : permPts) {
      if (!(ik instanceof ConstantKey) || ((ConstantKey<?>) ik).getValue() != null) return false;
    }
    return true;
  }

  /**
   * Resolves the {@code perm} argument to its constant, ordered list of axis indices.
   *
   * <p>Each constant list/tuple is an alternative permutation candidate. When the analysis sees
   * more than one distinct candidate, picking either would be unsound, so the method conservatively
   * returns {@code null} (⊤).
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param permPts The {@code perm} argument's points-to set.
   * @return The single resolved permutation, or {@code null} when any element is non-constant or
   *     there is more than one distinct candidate.
   */
  private List<Integer> resolvePermList(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> permPts) {
    Set<List<Integer>> candidates = new HashSet<>();
    for (InstanceKey ik : permPts) {
      Set<List<Dimension<?>>> lists;
      try {
        lists = this.getShapesFromShapeArgument(builder, Collections.singleton(ik));
      } catch (IllegalStateException e) {
        // `getShapesFromShapeArgument` throws for an unrecognized shape form; degrade that to ⊤.
        LOGGER.fine(
            () -> "Could not resolve perm of " + describe(this.getSource()) + ": " + e + ".");
        return null;
      }
      if (lists == null) return null;
      for (List<Dimension<?>> list : lists) {
        List<Integer> candidate = new ArrayList<>(list.size());
        for (Dimension<?> d : list) {
          if (!(d instanceof NumericDim)) return null;
          candidate.add(((NumericDim) d).value());
        }
        candidates.add(candidate);
      }
    }
    // Exactly one distinct candidate is the resolved permutation; zero or several is ⊤.
    return candidates.size() == 1 ? candidates.iterator().next() : null;
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
