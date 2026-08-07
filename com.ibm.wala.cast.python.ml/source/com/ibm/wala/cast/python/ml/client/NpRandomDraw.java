package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Shared base for the draws on NumPy's legacy module-level random surface, {@code np.random.randn}
 * and friends. Every draw returns a {@code float64} array of the requested shape &mdash; except
 * that a draw with no shape request returns a single Python {@code float} instead, which is why the
 * return type is argument-dependent rather than fixed. Subclasses differ only in how their
 * signature spells the shape request, and therefore in how they recognize its absence.
 *
 * <p>The no-request case matters more than its rarity suggests: {@code tf.Variable(rng.randn())} is
 * the corpus idiom, and TensorFlow's weak default converts a Python number to {@code float32}, not
 * to NumPy's {@code float64}. The dtype lattice has no element for "a Python number", so the case
 * is modeled as a rank-0 {@code float32}, which is the encoding the analysis already gives a Python
 * float literal ({@code TensorGenerator.getDTypesOfValue} reads one as {@code float32}) and is what
 * every TensorFlow consumer computes for the value. It does diverge from NumPy's own promotion,
 * which reads a Python float as {@code float64}: {@code np.array(np.random.randn())} is a {@code
 * float64} array at runtime and a {@code float32} one here. See <a
 * href="https://github.com/wala/ML/issues/827">wala/ML#827</a>.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/random/legacy.html">NumPy legacy random
 *     generation</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public abstract class NpRandomDraw extends TensorTypeAllocator {

  private static final Logger LOGGER = getLogger(NpRandomDraw.class.getName());

  public NpRandomDraw(PointsToSetVariable source) {
    super(source);
  }

  public NpRandomDraw(CGNode node) {
    super(node);
  }

  /**
   * Whether this call requests no shape, and so draws a single Python number rather than an array.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code true} iff the call makes no shape request.
   */
  protected abstract boolean isScalarDraw(PropagationCallGraphBuilder builder);

  /**
   * The shapes of an array draw, i.e. one whose call does make a shape request. Reached only when
   * the request did not already resolve through the inherited shape-argument machinery, so the
   * default is that base's ⊤.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The drawn array's shapes, or {@code null} (⊤) if the request does not resolve.
   */
  protected Set<List<Dimension<?>>> getArrayDrawShapes(PropagationCallGraphBuilder builder) {
    return super.getDefaultShapes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * <p>A draw with no shape request is a single Python number, so it is rank 0.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The singleton rank-0 shape for a scalar draw, else the array draw's shapes.
   */
  @Override
  protected final Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    if (this.isScalarDraw(builder)) {
      LOGGER.fine(
          () ->
              "No shape requested of "
                  + this.getSignature()
                  + " for source: "
                  + describe(this.getSource())
                  + "; the draw is a scalar.");
      return Set.of(List.of());
    }

    return this.getArrayDrawShapes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * <p>An array draw is {@code float64}, NumPy's floating-point default. A scalar draw is a Python
   * {@code float}, which this analysis encodes as {@code float32}; see the class comment for why
   * and for where that encoding diverges from NumPy's own promotion.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return A singleton set holding the drawn value's dtype.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.isScalarDraw(builder) ? EnumSet.of(FLOAT32) : EnumSet.of(FLOAT64);
  }

  /**
   * Returns the producing library of the modeled value: a {@code np.random} draw, so the value is
   * an ndarray or a Python number (wala/ML#724). Neither is a TensorFlow computation, which is the
   * distinction the origin exists to record.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
