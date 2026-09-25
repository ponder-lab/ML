package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code tf.unstack(value, num=None, axis=0, ...)}: every piece has the {@code value}
 * shape with the {@code axis} dimension removed, so the single modeled piece stands for all of
 * them, and the dtype is inherited from {@code value}. A non-constant {@code axis} leaves the
 * result ⊤, since the analysis cannot say which dimension goes. The summary used to return a list
 * whose first element was an unrelated constant and whose other elements were empty, so the first
 * unpacked value read a fabricated member and the second read nothing; the fabricated member then
 * reached every value composed from it.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/unstack">tf.unstack</a>
 */
public class Unstack extends PassThroughUnaryTensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Unstack.class.getName());

  public Unstack(PointsToSetVariable source) {
    super(source);
  }

  public Unstack(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "value";
  }

  /**
   * Derives each piece's shape from the {@code value} (arg 0) shape by removing the {@code axis}
   * (arg 2, default 0) dimension.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return The piece shapes, or {@code null} (⊤) when the input shape or a passed {@code axis} is
   *     unknown, or the {@code axis} is out of the shape's range.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null || inputShapes.isEmpty()) return inputShapes;

    // An absent axis defaults to 0; a passed one must resolve to a constant, since an empty
    // points-to set for a passed axis means a computed one.
    Integer axis;
    if (this.isAxisPassed(builder)) {
      axis = this.constantIntArgOrNull(builder, 2, "axis");
      if (axis == null) {
        LOGGER.fine(() -> "Non-constant axis for " + describe(this.getSource()) + "; returning ⊤.");
        return null;
      }
    } else axis = 0;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes) {
      int rank = input.size();
      if (rank == 0) return null; // Unstacking a scalar is a runtime error.
      int normalized = axis < 0 ? axis + rank : axis;
      if (normalized < 0 || normalized >= rank) return null;
      List<Dimension<?>> out = new ArrayList<>(input);
      out.remove(normalized);
      ret.add(out);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Whether the {@code axis} argument is passed at the call site, positionally (a fourth use beyond
   * the callable, {@code value}, and {@code num}) or as a resolvable keyword. A keyword-passed
   * <em>computed</em> axis is indistinguishable from an absent one here and is treated as absent.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return {@code true} iff some call site passes {@code axis}.
   */
  private boolean isAxisPassed(PropagationCallGraphBuilder builder) {
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) return call.getNumberOfPositionalParameters() > 3;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode()))
      if (callerInvoke.snd instanceof PythonInvokeInstruction
          && ((PythonInvokeInstruction) callerInvoke.snd).getNumberOfPositionalParameters() > 3)
        return true;
    OrdinalSet<InstanceKey> keywordPts = this.getArgumentPointsToSet(builder, 2, "axis");
    return keywordPts != null && !keywordPts.isEmpty();
  }

  /**
   * Resolves an argument to a constant integer via its points-to set.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param paramPos The argument's positional index, excluding {@code self}.
   * @param paramName The argument's keyword name.
   * @return The constant value, or {@code null} when the argument is absent, non-constant, or not
   *     an integer.
   */
  private Integer constantIntArgOrNull(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts == null || pts.isEmpty()) return null;
    Integer found = null;
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object value = ((ConstantKey<?>) ik).getValue();
      if (!(value instanceof Number)) return null;
      int intValue = ((Number) value).intValue();
      if (found != null && found != intValue) return null; // Ambiguous.
      found = intValue;
    }
    return found;
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes in {@link
   * #getDefaultShapes}, which the pass-through identity record path would bypass, so the record
   * view routes through the legacy transform.
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
