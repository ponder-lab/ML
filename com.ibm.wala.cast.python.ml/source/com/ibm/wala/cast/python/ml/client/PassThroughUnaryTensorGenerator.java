package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Common base for tensor generators that produce a fresh tensor with the same shape and dtype as a
 * single input argument. Covers element-wise unary math ops ({@code tf.math.abs}, {@code
 * tf.math.tanh}, {@code tf.math.exp}, etc.), pure pass-throughs ({@code tf.identity}, {@code
 * tf.stop_gradient}), and shape-preserving normalizers ({@code tf.math.l2_normalize}, {@code
 * tf.nn.log_softmax}). See wala/ML#449 (Tier 1 + Tier 2).
 *
 * <p>Subclasses identify the input argument via {@link #getInputParameterPosition()} and {@link
 * #getInputParameterName()}; everything else is shared. The arg-resolution itself uses the same
 * PTS-first-with-caller-walk-fallback pattern as {@link Sigmoid} (which predates this base and
 * still has its own copy — left untouched here to keep this change reviewable; refactoring {@code
 * Sigmoid}/{@code Identity}/{@code StopGradient} onto this base is a clean follow-up once {@code
 * #202} lands).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public abstract class PassThroughUnaryTensorGenerator extends TensorGenerator {

  /**
   * Constructs from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  protected PassThroughUnaryTensorGenerator(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  protected PassThroughUnaryTensorGenerator(CGNode node) {
    super(node);
  }

  /**
   * Positional index of the input argument, <em>excluding</em> the modeled {@code self} receiver.
   * Almost always {@code 0} (the first user-facing positional after {@code self}).
   *
   * @return The input arg's positional index.
   */
  protected abstract int getInputParameterPosition();

  /**
   * Keyword name of the input argument as declared in the XML (e.g. {@code "x"}, {@code "input"},
   * {@code "logits"}). Used for keyword-argument resolution when the call site uses kwargs.
   *
   * @return The input arg's keyword name.
   */
  protected abstract String getInputParameterName();

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return shapesOfArg(builder, getInputParameterPosition(), getInputParameterName());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> dtypes = dtypesOfArg(builder, getInputParameterPosition(), getInputParameterName());
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * PTS-first arg-shape resolver with caller-walk fallback. Mirrors {@link Sigmoid#shapesOfArg}.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolved shapes, or {@code null} if neither path recovers.
   */
  private Set<List<Dimension<?>>> shapesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, pts);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }
    return this.getArgumentShapesViaCallers(builder, paramPos, paramName);
  }

  /**
   * Dtype counterpart of {@link #shapesOfArg}.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolved dtypes, or {@code null} if neither path recovers.
   */
  private Set<DType> dtypesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, pts);
      if (dtypes != null && !dtypes.isEmpty()) return dtypes;
    }
    return this.getArgumentDTypesViaCallers(builder, paramPos, paramName);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
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
