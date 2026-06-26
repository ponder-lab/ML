package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FILL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the TensorFlow <code>fill()</code> function.
 *
 * <p>The fill() function creates a new tensor with a specified shape and fills it with a specified
 * value.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/fill">TensorFlow fill() API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Fill extends Constant {

  private static final Logger LOGGER = getLogger(Fill.class.getName());

  /** Canonical {@code tf.fill()} signature, reused in diagnostics. */
  private static final String SIGNATURE = TYPE_REFERENCE_TO_SIGNATURE.get(FILL.getDeclaringClass());

  protected enum Parameters {
    DIMS,
    VALUE,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  /**
   * The dtype argument is not explicitly provided to fill(); rather, the dtype is inferred from the
   * `value` argument.
   */
  private static final int VALUE_NUMBER_FOR_DTYPE_ARGUMENT = UNDEFINED_PARAMETER_POSITION;

  public Fill(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getDTypeArgumentValueNumber() {
    return VALUE_NUMBER_FOR_DTYPE_ARGUMENT;
  }

  @Override
  protected int getValueParameterPosition() {
    return Parameters.VALUE.getIndex();
  }

  @Override
  protected String getValueParameterName() {
    return Parameters.VALUE.getName();
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.DIMS.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.DIMS.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  /**
   * Recovers the shape from a {@code .shape} {@code dims} argument when possible, else returns ⊤ as
   * the non-aborting floor. {@code tf.fill(x.shape, v)} takes its shape from {@code x}, the same
   * case the standard allocators recover (<a href="https://github.com/wala/ML/issues/610">
   * wala/ML#610</a>, <a href="https://github.com/wala/ML/issues/604">wala/ML#604</a>); a
   * list-literal {@code dims} (e.g. {@code tf.fill([2, 3], v)}) already resolves via the inherited
   * shape-argument machinery before this method is reached.
   *
   * <p>Unlike the standard allocators, {@code tf.fill} has no default shape: {@code dims} is
   * mandatory. Falling through to the ⊤ floor therefore means one of two things, both ⊤ because a
   * static analysis must not abort over a single call site:
   *
   * <ul>
   *   <li>{@code dims} was supplied but is unresolvable (e.g. it flows from an unmodeled,
   *       content-dependent source such as {@code json.loads(...)}). This is the wala/ML#370
   *       recovery case, and is what actually triggered <a
   *       href="https://github.com/wala/ML/issues/606">wala/ML#606</a>.
   *   <li>{@code dims} was genuinely omitted: a malformed call ({@code tf.fill} raises {@code
   *       TypeError} at runtime). The original code threw {@link UnsupportedOperationException}
   *       here to document that {@code dims} is mandatory; that aborted the whole analysis
   *       (wala/ML#606), so it is now a {@code FINE} marker instead.
   * </ul>
   *
   * <p>Mirrors the ⊤ floor {@code TensorTypeAllocator} took for the standard allocators (<a
   * href="https://github.com/wala/ML/issues/604">wala/ML#604</a>); {@code Fill} extends {@code
   * Constant} rather than {@code TensorTypeAllocator}, so it carries the floor here.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code null} (⊤, unknown shape); never an empty set, since the allocation is a tensor.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Recovery: tf.fill(x.shape, v) takes its shape from x, the same case the standard allocators
    // recover (wala/ML#604). Resolve it rather than dropping to ⊤. wala/ML#610.
    Set<List<Dimension<?>>> fromShapeAttribute =
        this.getShapeFromShapeAttributeArgument(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());
    if (fromShapeAttribute != null && !fromShapeAttribute.isEmpty()) {
      LOGGER.fine(
          "Recovered "
              + SIGNATURE
              + " shape from a .shape argument for source: "
              + source
              + " -> "
              + fromShapeAttribute
              + ".");
      return fromShapeAttribute;
    }
    boolean dimsSupplied =
        this.isKeywordArgumentPresent(builder, Parameters.DIMS.getName())
            || this.getNumberOfPossiblePositionalArguments(builder).stream()
                .anyMatch(n -> n >= Parameters.DIMS.getIndex() + 1);
    LOGGER.fine(
        dimsSupplied
            ? "Could not resolve the dims argument of "
                + SIGNATURE
                + " for source: "
                + source
                + "; returning ⊤. Candidate for a wala/ML#370 shape annotation."
            : SIGNATURE
                + " reached without its mandatory dims argument for source: "
                + source
                + " (malformed call?); returning ⊤.");
    return null;
  }
}
