package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Locale;
import java.util.logging.Logger;

/**
 * Generator for the {@code np.random} draws that spell their shape as a {@code size} argument,
 * {@code np.random.normal(loc, scale, size)} and {@code np.random.uniform(low, high, size)}. The
 * two distributions share a signature shape &mdash; two distribution parameters, then {@code size}
 * &mdash; so one generator serves both; only the parameters' names differ, and those name nothing
 * this analysis reads.
 *
 * <p>The {@code size} argument is an ordinary shape argument (an integer or a tuple of them), so
 * the inherited {@link TensorTypeAllocator} machinery resolves it. Omitting it, or passing {@code
 * None}, is the scalar draw {@link NpRandomDraw} describes.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html">
 *     numpy.random.normal()</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpRandomSizedDraw extends NpRandomDraw {

  private static final Logger LOGGER = getLogger(NpRandomSizedDraw.class.getName());

  protected enum Parameters {
    LOC,
    SCALE,
    SIZE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public NpRandomSizedDraw(PointsToSetVariable source) {
    super(source);
  }

  public NpRandomSizedDraw(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * <p>The draw is a scalar when the call omits {@code size} altogether or passes an explicit
   * {@code None}, which NumPy treats identically. A {@code size} that is present but unresolvable
   * is not evidence of either, so it reads as an array draw and floors to ⊤.
   *
   * <p>Whether the argument was passed is read off the call sites rather than off the callee
   * frame's parameter. A manual anchoring has the parameter regardless, since the summary declares
   * it, so its value number is present even for a call that supplied nothing; the call sites are
   * what distinguish the two, and they are visible under both anchorings.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code true} iff the call requests no size.
   */
  @Override
  protected boolean isScalarDraw(PropagationCallGraphBuilder builder) {
    int sizePosition = this.getShapeParameterPosition();
    boolean passed =
        this.getNumberOfPossiblePositionalArguments(builder).stream()
                .anyMatch(count -> count > sizePosition)
            || this.isKeywordArgumentPresent(builder, this.getShapeParameterName());

    if (!passed) {
      LOGGER.fine(
          () ->
              "No size argument for source: "
                  + describe(this.getSource())
                  + "; the draw is a scalar.");
      return true;
    }

    OrdinalSet<InstanceKey> sizePointsToSet =
        this.getArgumentPointsToSet(builder, sizePosition, this.getShapeParameterName());

    if (allNullConstants(sizePointsToSet)) {
      LOGGER.fine(
          () ->
              "Explicit `None` size for source: "
                  + describe(this.getSource())
                  + "; the draw is a scalar.");
      return true;
    }

    return false;
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SIZE.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SIZE.getName();
  }
}
