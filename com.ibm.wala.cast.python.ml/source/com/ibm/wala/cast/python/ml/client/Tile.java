package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
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
 * Generator for {@code tf.tile(input, multiples, name=None)}. Output dtype is inherited from the
 * {@code input}. Output shape multiplies each axis by the corresponding entry of {@code multiples},
 * so a {@code (M, N)} input with {@code multiples = [a, b]} yields {@code (a*M, b*N)}. A
 * non-constant {@code multiples}, a {@code multiples} whose length differs from the input rank, or
 * an input axis that is not statically numeric, falls back to ⊤ rather than risk an unsound shape.
 * Previously modeled as a first-argument {@code pass_through}, which reported the input shape
 * unchanged. See <a href="https://github.com/wala/ML/issues/513">wala/ML#513</a> bucket 2a.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/tile">tf.tile</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Tile extends PassThroughUnaryTensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Tile.class.getName());

  /**
   * Constructs from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public Tile(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public Tile(CGNode node) {
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
   * Derives the output shape by multiplying each {@code input} (arg 0) axis by the corresponding
   * entry of the {@code multiples} (arg 1) argument.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when {@code input}'s shape is
   *     unknown, {@code multiples} is a non-constant, its length differs from the input rank, or an
   *     input axis is not statically numeric for some input alternative.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null) return null;

    List<Integer> multiples =
        resolveMultiplesList(builder, this.getArgumentPointsToSet(builder, 1, "multiples"));
    if (multiples == null) {
      LOGGER.fine(
          () ->
              "Non-constant multiples for "
                  + Loggables.describe(this.getSource())
                  + "; returning ⊤.");
      return null;
    }

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes) {
      List<Dimension<?>> out = tileShape(input, multiples);
      // A ⊤ (null) for any input alternative joins to ⊤ for the whole result.
      if (out == null) return null;
      ret.add(out);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Tiles a single input shape, multiplying axis {@code i} by {@code multiples[i]}.
   *
   * @param input The input shape.
   * @param multiples The per-axis tiling counts.
   * @return The tiled shape, or {@code null} (⊤) when {@code multiples}'s length differs from the
   *     input rank or an input axis is not statically numeric.
   */
  private static List<Dimension<?>> tileShape(List<Dimension<?>> input, List<Integer> multiples) {
    if (multiples.size() != input.size()) return null;
    List<Dimension<?>> out = new ArrayList<>(input.size());
    for (int i = 0; i < input.size(); i++) {
      Dimension<?> dim = input.get(i);
      if (!(dim instanceof NumericDim)) return null;
      out.add(new NumericDim(((NumericDim) dim).value() * multiples.get(i)));
    }
    return out;
  }

  /**
   * Resolves the {@code multiples} argument to its constant, ordered list of per-axis counts.
   *
   * <p>Each constant list/tuple is an alternative candidate. When the analysis sees more than one
   * distinct candidate, picking either would be unsound, so the method conservatively returns
   * {@code null} (⊤).
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param multiplesPts The {@code multiples} argument's points-to set.
   * @return The single resolved list of counts, or {@code null} when {@code multiples} is absent or
   *     any element is non-constant, or there is more than one distinct candidate.
   */
  private List<Integer> resolveMultiplesList(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> multiplesPts) {
    if (multiplesPts == null || multiplesPts.isEmpty()) return null;
    Set<List<Integer>> candidates = new HashSet<>();
    for (InstanceKey ik : multiplesPts) {
      Set<List<Dimension<?>>> lists;
      try {
        lists = this.getShapesFromShapeArgument(builder, Collections.singleton(ik));
      } catch (IllegalStateException e) {
        // `getShapesFromShapeArgument` throws for an unrecognized shape form; degrade that to ⊤.
        LOGGER.fine(
            () ->
                "Could not resolve multiples of "
                    + Loggables.describe(this.getSource())
                    + ": "
                    + e
                    + ".");
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
    // Exactly one distinct candidate is the resolved list; zero or several is ⊤.
    return candidates.size() == 1 ? candidates.iterator().next() : null;
  }
}
