package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for the {@code np.random} draws that spell their shape as variadic dimensions, {@code
 * np.random.randn(d0, d1, ...)} and {@code np.random.rand(d0, d1, ...)}. The call's arity is the
 * drawn array's rank, so a call with no arguments is the scalar draw {@link NpRandomDraw}
 * describes.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html">
 *     numpy.random.randn()</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpRandomVariadicDraw extends NpRandomDraw {

  private static final Logger LOGGER = getLogger(NpRandomVariadicDraw.class.getName());

  public NpRandomVariadicDraw(PointsToSetVariable source) {
    super(source);
  }

  public NpRandomVariadicDraw(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Every dimension is variadic, so the draw is a scalar exactly when the call passes nothing.
   * An arity that does not resolve to a single count &mdash; no call site was recovered, or several
   * disagree &mdash; is not evidence of an empty call, so it reads as an array draw and floors to ⊤
   * downstream.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code true} iff the call passes no arguments.
   */
  @Override
  protected boolean isScalarDraw(PropagationCallGraphBuilder builder) {
    Set<Integer> arities = this.getNumberOfPossiblePositionalArguments(builder);
    return arities.size() == 1 && arities.contains(0);
  }

  /**
   * {@inheritDoc}
   *
   * <p>One axis per argument, each sized by that argument's integer value. A dimension the analysis
   * cannot fold is {@link UnresolvedDim} rather than {@link
   * com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim}: the argument is a Python scalar, so
   * the runtime size is a fixed integer this analysis could not compute rather than one the runtime
   * reports as {@code None} (wala/ML#721). The rank is therefore recovered even when no size is.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The singleton drawn shape, or {@code null} (⊤) when the arity does not resolve.
   */
  @Override
  protected Set<List<Dimension<?>>> getArrayDrawShapes(PropagationCallGraphBuilder builder) {
    Set<Integer> arities = this.getNumberOfPossiblePositionalArguments(builder);

    if (arities.size() != 1) {
      LOGGER.fine(
          () ->
              "Ambiguous arity "
                  + arities
                  + " for source: "
                  + describe(this.getSource())
                  + "; returning ⊤ (unknown shape).");
      return null;
    }

    int rank = arities.iterator().next();
    List<Dimension<?>> shape = new ArrayList<>(rank);

    for (int axis = 0; axis < rank; axis++)
      shape.add(axisDim(this.getPossibleArgumentValues(builder, axis, null)));

    LOGGER.fine(
        () -> "Recovered shape " + shape + " for source: " + describe(this.getSource()) + ".");
    return Set.of(shape);
  }

  /**
   * Maps an axis argument's possible values to a dimension.
   *
   * @param values The argument's possible integer values, as {@link
   *     TensorTypeAllocator#getPossibleArgumentValues} reports them.
   * @return A {@link NumericDim} when the argument folds to exactly one integer, else {@link
   *     UnresolvedDim#INSTANCE}.
   */
  private static Dimension<?> axisDim(Set<Optional<Integer>> values) {
    if (values.size() == 1) {
      Optional<Integer> only = values.iterator().next();
      if (only.isPresent()) return new NumericDim(only.get());
    }

    return UnresolvedDim.INSTANCE;
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }
}
